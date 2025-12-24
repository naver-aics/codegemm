# CodeGEMM: A Codebook-Centric Approach to Efficient GEMM in Quantized LLMs (NeurIPS 2025)

This repository provides an official implementation of [CodeGEMM](https://arxiv.org/abs/2512.17970).

> **Note**: This codebase is based on the [AQLM repository](https://github.com/Vahe1994/AQLM/tree/main), which provides the official PyTorch implementation for Extreme Compression of Large Language Models via Additive Quantization.

## 🔧 Installation

- NGC image used: `nvcr.io/nvidia/pytorch:24.10-py3` ([details / release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-10.html))

Install the Python package in editable mode:

```bash
pip install -e .
```

To install the custom CUDA kernel for Psumbook-based GEMV used in CodeGEMM:

```bash
cd codegemm/inference/custom_kernel
source do_install.sh
```

## Quick Evaluation

Get started quickly by downloading a pre-quantized model and running evaluation.

### Download checkpoint

Available checkpoints:
- [Llama-3.1-8B-Instruct-Codegemm-m2v8g128](https://huggingface.co/gunho1123/Llama-3.1-8B-Instruct-Codegemm-m2v8g128)
- [Llama-3.1-8B-Instruct-Codegemm-m2v8g128-PVtuning](https://huggingface.co/gunho1123/Llama-3.1-8B-Instruct-Codegemm-m2v8g128-PVtuning)

```bash
REPO_ID=gunho1123/Llama-3.1-8B-Instruct-Codegemm-m2v8g128

python download_model.py --repo_id $REPO_ID --local_dir $REPO_ID
```

### Evaluation
```bash
REPO_ID=gunho1123/Llama-3.1-8B-Instruct-Codegemm-m2v8g128

# MMLU
python run_eval.py --model_path $REPO_ID

# CSR
python run_eval.py --model_path $REPO_ID --downstream
```


## 🚀 PTQ (Post-Training Quantization)

### Quantization

```bash
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
CACHE_DIR=hf_models

VAL_SIZE=256
NUM_CALIBRATION_SEQUENCES=2048
SEQLEN=8192
NUM_CODEBOOKS=2
IN_GROUP_SIZE=8
SCALE_GROUP_SIZE=128
SAVE_PATH=./quantized/$MODEL_PATH-m$NUM_CODEBOOKS-b8-v$IN_GROUP_SIZE-g$SCALE_GROUP_SIZE-$NUM_CALIBRATION_SEQUENCES-$SEQLEN

mkdir -p $SAVE_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 python run_quant.py $MODEL_PATH pajama \
    --cache_dir $CACHE_DIR \
    --nsamples=$NUM_CALIBRATION_SEQUENCES \
    --model_seqlen=$SEQLEN \
    --val_size=$VAL_SIZE \
    --num_codebooks=$NUM_CODEBOOKS \
    --nbits_per_codebook=8 \
    --in_group_size=$IN_GROUP_SIZE \
    --scale_group_size=$SCALE_GROUP_SIZE \
    --lr 1e-4 \
    --finetune_lr 1e-5 \
    --relative_mse_tolerance=0.01 \
    --finetune_batch_size=32 \
    --finetune_max_epochs=10 \
    --finetune_early_stop=3 \
    --finetune_keep_best \
    --local_batch_size=2 \
    --offload_activations \
    --resume \
    --trust_remote_code \
    --seed 0 \
    --save $SAVE_PATH
```

### Evaluation

To evaluate after PTQ, first pack the quantized model and then run evaluation:

```bash
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
CACHE_DIR=hf_models

NUM_CALIBRATION_SEQUENCES=2048
SEQLEN=8192
NUM_CODEBOOKS=2
IN_GROUP_SIZE=8
SCALE_GROUP_SIZE=128
SAVE_PATH=$MODEL_PATH-m$NUM_CODEBOOKS-b8-v$IN_GROUP_SIZE-g$SCALE_GROUP_SIZE-$NUM_CALIBRATION_SEQUENCES-$SEQLEN

mkdir -p packed_hf

python run_convert.py --model $MODEL_PATH \
    --in_path quantized/$SAVE_PATH \
    --out_path packed_hf/$SAVE_PATH \
    --save_safetensors --load_model --swap_in_place --cache_dir $CACHE_DIR

# MMLU
python run_eval.py --model_path packed_hf/$SAVE_PATH

# CSR
python run_eval.py --model_path packed_hf/$SAVE_PATH --downstream
```

## 🔄 PV-Tuning

PV-Tuning is a method for improving quantized model accuracy beyond straight-through estimation. For more details, see the [paper](https://arxiv.org/abs/2405.14852).

### Prepare Dataset
```bash
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
CACHE_DIR=hf_models

NUM_CALIBRATION_SEQUENCES=2048
SEQLEN=8192
TOKENIZED_DATASET_PATH=redpajama_tokenized_llama3_8192
DATASET=DKYoon/SlimPajama-6B
DATASET_CONFIG_NAME=default

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 torchrun --master-port 3456 --nproc-per-node=1 run_finetune.py \
    --base_model $MODEL_PATH \
    --quantized_model ./doesnt_matter \
    --block_type LlamaDecoderLayer \
    --dataset_name=$DATASET \
    --split train \
    --dataset_config_name $DATASET_CONFIG_NAME \
    --cache_dir=$CACHE_DIR \
    --trust_remote_code \
    --model_seqlen=$SEQLEN \
    --preprocessing_num_workers=64 \
    --preprocessing_chunk_length 100000 \
    --save_dataset_and_exit $TOKENIZED_DATASET_PATH
```

### Finetune

```bash
NUM_GPUS=4

MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
CACHE_DIR=hf_models

NUM_CALIBRATION_SEQUENCES=2048
SEQLEN=8192
NUM_CODEBOOKS=2
IN_GROUP_SIZE=8
SCALE_GROUP_SIZE=128

SAVE_PATH=$MODEL_PATH-m$NUM_CODEBOOKS-b8-v$IN_GROUP_SIZE-g$SCALE_GROUP_SIZE-$NUM_CALIBRATION_SEQUENCES-$SEQLEN
TOKENIZED_DATASET_PATH=redpajama_tokenized_llama3_8192

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=$NUM_GPUS --master_port=29501 run_finetune.py \
    --base_model $MODEL_PATH \
    --quantized_model quantized/$SAVE_PATH \
    --model_seqlen=$SEQLEN \
    --block_type LlamaDecoderLayer \
    --load_dtype bfloat16 \
    --amp_dtype bfloat16 \
    --code_dtype uint16 \
    --dataset_name=$TOKENIZED_DATASET_PATH \
    --split none \
    --seed 42 \
    --preprocessing_chunk_length 100000 \
    --cache_dir=$CACHE_DIR \
    --trust_remote_code \
    --update_codes \
    --update_codebooks_and_scales \
    --update_non_quantized_parameters \
    --lamb \
    --debias \
    --lr 3e-4 \
    --adam_beta1 0.90 \
    --adam_beta2 0.95 \
    --max_code_change_per_step 1e-2 \
    --code_lr 1e-2 \
    --code_beta1 0.0 \
    --code_beta2 0.95 \
    --beam_size 5 \
    --delta_decay 0 \
    --batch_size=128 \
    --microbatch_size=1 \
    --max_epochs 1 \
    --gradient_checkpointing \
    --print_every_steps=1 \
    --verbose_optimizer \
    --eval_every_steps=10 \
    --keep_best_model \
    --save finetuned/$SAVE_PATH \
    --save_every_steps 100 \
    --attn_implementation flash_attention_2
```

### Evaluation

To evaluate after finetuning, first convert and pack the model, then run evaluation:

**Step 1: Convert legacy model format**

```bash
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct

NUM_CALIBRATION_SEQUENCES=2048
SEQLEN=8192
NUM_CODEBOOKS=2
IN_GROUP_SIZE=8
SCALE_GROUP_SIZE=128

SAVE_PATH=$MODEL_PATH-m$NUM_CODEBOOKS-b8-v$IN_GROUP_SIZE-g$SCALE_GROUP_SIZE-$NUM_CALIBRATION_SEQUENCES-$SEQLEN
CACHE_DIR=hf_models

CUDA_VISIBLE_DEVICES=0 python codegemm/quantization/convert_legacy_model_format.py \
    --base_model $MODEL_PATH \
    --quantized_model quantized/$SAVE_PATH \
    --pv_fsdp_dir finetuned/$SAVE_PATH/best_model \
    --monkeypatch_old_pickle \
    --save converted/$SAVE_PATH \
    --cache_dir $CACHE_DIR

cp quantized/$SAVE_PATH/args.pt converted/$SAVE_PATH
```

**Step 2: Pack and evaluate**

```bash
MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct
CACHE_DIR=hf_models

NUM_CALIBRATION_SEQUENCES=2048
SEQLEN=8192
NUM_CODEBOOKS=2
IN_GROUP_SIZE=8
SCALE_GROUP_SIZE=128

SAVE_PATH=$MODEL_PATH-m$NUM_CODEBOOKS-b8-v$IN_GROUP_SIZE-g$SCALE_GROUP_SIZE-$NUM_CALIBRATION_SEQUENCES-$SEQLEN

python run_convert.py --model converted/$MODEL_PATH \
    --in_path quantized/$SAVE_PATH \
    --out_path packed_hf/$SAVE_PATH-pvtune \
    --save_safetensors --load_model --swap_in_place --cache_dir $CACHE_DIR

# MMLU
python run_eval.py --model_path packed_hf/$SAVE_PATH-pvtune

# CSR
python run_eval.py --model_path packed_hf/$SAVE_PATH-pvtune --downstream
```

## 📜 Citation

If you find **CodeGEMM** useful, please cite:

```bibtex
@inproceedings{park2025codegemm,
  title={CodeGEMM: A Codebook-Centric Approach to Efficient GEMM in Quantized LLMs},
  author={Park, Gunho and Bae, Jeongin and Kim, Byeongwook and Ryu, Jiwon and Kim, Hoseung and Kwon, Se Jung and Lee, Dongsoo and others},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year      = {2025}
}
```

## ⚡ Throughput Evaluation

To evaluate the inference throughput of CodeGEMM, use the provided benchmark script:

```bash
cd codegemm/inference
source do_tps.sh
```

## 📄 License

```
CodeGEMM
Copyright (c) 2025-present NAVER Cloud Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
