# Enter the inference folder

# (Optional) choose GPU
export GPU=0

# model_name_or_path="meta-llama/Llama-3.1-8B"
model_name_or_path="meta-llama/Llama-3.1-70B-Instruct"

# ---------------------------
# CodeGEMM backend
# ---------------------------

backend="codegemm"
vec_len=4
group_size=32
num_codebooks=1
nbits_per_codebook=8

CUDA_VISIBLE_DEVICES=${GPU} python generate.py --compile 2 --num_samples 5 \
  --model_name "${model_name_or_path}" --vec_len ${vec_len} \
  --group_size ${group_size} --num_codebooks ${num_codebooks} \
  --nbits_per_codebook ${nbits_per_codebook} --dtype "float16" \
  --backend ${backend} --max_new_tokens 100 --random_init

backend="codegemm"
vec_len=8
group_size=128
num_codebooks=2
nbits_per_codebook=8

CUDA_VISIBLE_DEVICES=${GPU} python generate.py --compile 2 --num_samples 5 \
  --model_name "${model_name_or_path}" --vec_len ${vec_len} \
  --group_size ${group_size} --num_codebooks ${num_codebooks} \
  --nbits_per_codebook ${nbits_per_codebook} --dtype "float16" \
  --backend ${backend} --max_new_tokens 100 --random_init

# ---------------------------
# AQLM backend
# ---------------------------

backend="aqlm"
vec_len=8
num_codebooks=2
nbits_per_codebook=8

CUDA_VISIBLE_DEVICES=${GPU} python generate.py --compile 2 --num_samples 5 \
  --model_name "${model_name_or_path}" --vec_len ${vec_len} \
  --group_size ${group_size} --num_codebooks ${num_codebooks} \
  --nbits_per_codebook ${nbits_per_codebook} --dtype "float16" \
  --backend ${backend} --max_new_tokens 100 --random_init



backend="aqlm"
vec_len=8
num_codebooks=1
nbits_per_codebook=16

CUDA_VISIBLE_DEVICES=${GPU} python generate.py --compile 2 --num_samples 5 \
  --model_name "${model_name_or_path}" --vec_len ${vec_len} \
  --group_size ${group_size} --num_codebooks ${num_codebooks} \
  --nbits_per_codebook ${nbits_per_codebook} --dtype "float16" \
  --backend ${backend} --max_new_tokens 100 --random_init
