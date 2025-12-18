# Originally from https://github.com/snu-mllab/GuidedQuant/blob/main/inference/plugin.py

import torch
import codegemm_kernel


@torch.library.custom_op("plugin::codegemm_gemv", mutates_args={"output"})
def codegemm_gemv(
    x: torch.Tensor, 
    output: torch.Tensor, 
    q_weight: torch.Tensor, 
    alpha: torch.Tensor, 
    codebook: torch.Tensor, 
    group_size: int) -> None:
    codegemm_kernel.codegemm_gemv(
        x, output, q_weight, alpha, codebook, group_size)

def codegemm_dequant(
    q_weight: torch.Tensor, 
    alpha: torch.Tensor, 
    codebook: torch.Tensor, 
    group_size: int) -> None:
    weight = codegemm_kernel.codegemm_dequant(
        q_weight, alpha, codebook, group_size)
    return weight

@codegemm_gemv.register_fake
def _(x, output, q_weight, alpha, codebook, group_size):
    return None