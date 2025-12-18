import torch
import torch.nn as nn
import sys, os

import math
from typing import Optional

import torch
import torch.nn as nn

from codegemm.inference.codegemm_linear import CodeGEMMLinear


def get_num_layers(config) -> int:
    match config.model_type:
        case "llama" | "mistral" | "mixtral" | "gemma" | "phi3" | "qwen2":
            return config.num_hidden_layers
        case unknown_type:
            raise NotImplementedError(f"Can't get number of layers for {unknown_type}")

def get_module_by_name(model, target_name):
    """
    Get a module from the model by its name.
    
    Args:
        model: PyTorch model
        target_name: Full module name (e.g., "model.layers.0.self_attn.q_proj")
    
    Returns:
        The module if found, None otherwise
    
    Example:
        >>> module = get_module_by_name(model, "model.layers.0.self_attn.q_proj")
    """
    # Method 1: Using named_modules() - iterates through all modules
    for name, module in model.named_modules():
        if name == target_name:
            return module
    return None


def get_module_by_name_fast(model, target_name):
    """
    Get a module from the model by its name (faster version).
    Uses get_submodule which is more efficient than iterating.
    
    Args:
        model: PyTorch model
        target_name: Full module name (e.g., "model.layers.0.self_attn.q_proj")
    
    Returns:
        The module if found, None otherwise
    
    Example:
        >>> module = get_module_by_name_fast(model, "model.layers.0.self_attn.q_proj")
    """
    try:
        return model.get_submodule(target_name)
    except (AttributeError, KeyError):
        return None



def swap_quant_model(
    model, 
    config, 
    in_path,
    metadata: dict = None,
    layers_to_replace: list = None,
):
    """
    Replace nn.Linear layers with CodeGEMMLinear using quantized weights from checkpoint.
    
    Args:
        model: The model to modify
        config: Model config
        in_path: Path to directory containing {i}.pth files
        metadata: Quantization metadata (nbits_per_codebook, num_codebooks, etc.)
        layers_to_replace: List of layer name patterns to replace (e.g., ['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    """
    # Helper functions from LoRA
    def _get_submodules(model, key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(parent_module, child_name, new_module):
        setattr(parent_module, child_name, new_module)
    
    # Get metadata if not provided
    if metadata is None:
        metadata_path = os.path.join(in_path, "args.pt")
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path)
        else:
            raise ValueError(f"Metadata not found at {metadata_path}")
    
    # Default layers to replace
    if layers_to_replace is None:
        layers_to_replace = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    
    num_layers = get_num_layers(config)
    layers_prefix = get_layers_prefix(config)
    
    print(f"Starting to swap nn.Linear with CodeGEMMLinear for {num_layers} layers...")
    
    with torch.no_grad():
        for layer_idx in range(num_layers):
            # Load quantized layer data
            layer_path = os.path.join(in_path, f"{layer_idx}.pth")
            if not os.path.exists(layer_path):
                print(f"Warning: {layer_path} not found, skipping layer {layer_idx}")
                continue
            
            
            quantized_layer = torch.load(layer_path)
            print(f"\nProcessing layer {layer_idx}...")
            
            # Build a mapping of submodule names to their parameters
            # e.g., 'self_attn.q_proj' -> {codes, codebooks, scales, bias}
            param_dict = {}
            non_quantized_params = {}  # For LayerNorm, etc.
            
            for param_name, param_data in quantized_layer.named_parameters():
                # Parse parameter name: e.g., "self_attn.q_proj.quantized_weight.codes"
                parts = param_name.split('.')
                if len(parts) >= 3:
                    # Quantized parameters (e.g., "self_attn.q_proj.quantized_weight.codes")
                    submodule_name = '.'.join(parts[:-2])  # e.g., "self_attn.q_proj"
                    param_type = parts[-1]  # e.g., "codes", "codebooks", "scales", "bias"
                    
                    if submodule_name not in param_dict:
                        param_dict[submodule_name] = {}
                    param_dict[submodule_name][param_type] = param_data
                else:
                    # Non-quantized parameters (e.g., 'input_layernorm.weight', 'post_attention_layernorm.weight')
                    non_quantized_params[param_name] = param_data
            
            # First, load non-quantized parameters
            if non_quantized_params:
                print(f"  Loading {len(non_quantized_params)} non-quantized parameters...")
                for param_name, param_data in non_quantized_params.items():
                    full_name = f"{layers_prefix}.{layer_idx}.{param_name}"
                    try:
                        # Get the module that contains this parameter
                        module_name = '.'.join(param_name.split('.')[:-1])
                        param_attr = param_name.split('.')[-1]
                        
                        if module_name:
                            # e.g., "input_layernorm.weight" -> module="input_layernorm", attr="weight"
                            full_module_name = f"{layers_prefix}.{layer_idx}.{module_name}"
                            module = get_module_by_name_fast(model, full_module_name)
                            if module is not None and hasattr(module, param_attr):
                                getattr(module, param_attr).data = param_data
                                print(f"    Loaded {full_name}")
                            else:
                                print(f"    Warning: Could not find {full_module_name}.{param_attr}")
                        else:
                            # Direct parameter (unlikely in transformer layers)
                            target_module = get_module_by_name_fast(model, f"{layers_prefix}.{layer_idx}")
                            if target_module is not None and hasattr(target_module, param_name):
                                getattr(target_module, param_name).data = param_data
                                print(f"    Loaded {full_name}")
                    except Exception as e:
                        print(f"    Warning: Failed to load {full_name}: {e}")
            
            # Replace each nn.Linear in this layer
            for submodule_name, params in param_dict.items():
                # Check if this is a layer we want to replace
                should_replace = any(layer_type in submodule_name for layer_type in layers_to_replace)
                if not should_replace:
                    continue
                
                # Construct full module name
                full_name = f"{layers_prefix}.{layer_idx}.{submodule_name}"
                
                # Get the original module
                try:
                    parent, target, target_name = _get_submodules(model, full_name)
                except Exception as e:
                    print(f"Warning: Could not find module {full_name}: {e}")
                    continue
                
                # Check if it's a Linear layer
                if not isinstance(target, nn.Linear):
                    continue
                
                # Extract parameters for CodeGEMMLinear
                codes = params.get('codes')
                codebooks = params.get('codebooks')
                scales = params.get('scales')
                bias = params.get('bias', None)
                
                if codes is None or codebooks is None or scales is None:
                    print(f"Warning: Missing parameters for {full_name}, skipping")
                    continue
                
                # Get dimensions from original Linear layer
                in_features = target.in_features
                out_features = target.out_features
                
                print(metadata)
                # Create CodeGEMMLinear
                new_module = CodeGEMMLinear(
                    in_features=in_features,
                    out_features=out_features,
                    vec_len=metadata['in_group_size'],
                    group_size=metadata['scale_group_size'],
                    # out_group_size=metadata['out_group_size'],
                    num_codebooks=metadata['num_codebooks'],
                    nbits_per_codebook=metadata['nbits_per_codebook'],
                    bias=(bias is not None),
                    dtype=torch.half,
                )
                
                # Load quantized parameters
                temp = pack_int8_to_int32(codes.transpose(0,-1), msb_first=False)
                new_module.codes.data = temp.contiguous()
                new_module.codebooks.data = codebooks.squeeze(-2).contiguous()
                new_module.scales.data = scales.squeeze().transpose(0,-1).contiguous()
                if bias is not None:
                    new_module.bias.data = bias
                
                # Replace the module
                _replace_module(parent, target_name, new_module)
                print(f"  Replaced {full_name}: {in_features} -> {out_features}")
                
                # Clean up
                del target
            # Clean up loaded layer
            del quantized_layer
            torch.cuda.empty_cache()
    
    print("\nSwap completed!")
    return model


def get_layers_prefix(config) -> str:
    """Get the prefix for transformer layers based on model type."""
    match config.model_type:
        case "llama" | "mistral" | "mixtral" | "gemma" | "phi3" | "qwen2":
            return "model.layers"
        case unknown_type:
            raise NotImplementedError(f"Can't get layers prefix for {unknown_type}")

def pack_int8_to_int32(data: torch.Tensor, msb_first: bool = True) -> torch.Tensor:
    """
    Pack int8 tensor into int32 by grouping every 4 int8 values into one int32.
    Packs along the second-to-last dimension (axis=-2).
    
    Args:
        data: Input tensor with integer values. Can be any integer dtype (int8, int32, etc.)
              but each value must fit within 8 bits [-128, 127] or [0, 255].
              The second-to-last dimension must be divisible by 4.
              Shape: [..., k, n] where k % 4 == 0
        msb_first: If True, pack from MSB to LSB (big-endian style).
                   If False, pack from LSB to MSB (little-endian style).
                   
    Returns:
        Packed tensor with int32 dtype. Shape: [..., k/4, n]
        
    Example:
        >>> # 2D tensor: [k, n] -> [k/4, n]
        >>> data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.int8)
        >>> data.shape
        torch.Size([4, 2])
        >>> packed = pack_int8_to_int32(data, msb_first=True)
        >>> packed.shape
        torch.Size([1, 2])
        
        >>> # 3D tensor: [m, k, n] -> [m, k/4, n]
        >>> data = torch.randn(10, 128, 256).to(torch.int8)
        >>> packed = pack_int8_to_int32(data, msb_first=True)
        >>> packed.shape
        torch.Size([10, 32, 256])
    """
    import torch
    
    # Get the shape and ensure second-to-last dimension is divisible by 4
    original_shape = data.shape
    if len(original_shape) < 2:
        raise ValueError(f"Input tensor must have at least 2 dimensions, got {len(original_shape)}")
    
    k_dim = original_shape[-2]
    if k_dim % 4 != 0:
        raise ValueError(f"Second-to-last dimension (k={k_dim}) must be divisible by 4")
    
    # Reshape to (..., k/4, 4, n)
    new_shape = list(original_shape[:-2]) + [k_dim // 4, 4, original_shape[-1]]
    data_reshaped = data.reshape(new_shape)
    
    # Extract lower 8 bits and convert to uint8 for bit operations
    # Using bitwise AND to handle any integer dtype
    data_uint8 = (data_reshaped & 0xFF).to(torch.uint8)
    
    if msb_first:
        # MSB-first: [byte0, byte1, byte2, byte3] -> byte0<<24 | byte1<<16 | byte2<<8 | byte3
        # Pack along axis=-2 (the 4 elements)
        packed = (
            data_uint8[..., 0, :].to(torch.int32) << 24 |
            data_uint8[..., 1, :].to(torch.int32) << 16 |
            data_uint8[..., 2, :].to(torch.int32) << 8 |
            data_uint8[..., 3, :].to(torch.int32)
        )
    else:
        # LSB-first: [byte0, byte1, byte2, byte3] -> byte3<<24 | byte2<<16 | byte1<<8 | byte0
        packed = (
            data_uint8[..., 3, :].to(torch.int32) << 24 |
            data_uint8[..., 2, :].to(torch.int32) << 16 |
            data_uint8[..., 1, :].to(torch.int32) << 8 |
            data_uint8[..., 0, :].to(torch.int32)
        )
    
    return packed