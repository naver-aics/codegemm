/*
CodeGEMM
Copyright (c) 2025-present NAVER Cloud Corp.
Apache-2.0
*/

#ifndef CODEGEMM_CUH
#define CODEGEMM_CUH

#include <cassert>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include "datatype.h"
#include "typetraits.h"

#include <torch/extension.h>
#include <cuda_runtime.h>

template <int num_codebook, int len_vector>
__global__ void codegemm(
    const uint32_t* q_weight, // q_weight[num_codebook][K/len_vector/4][M]
    const __half* alpha, // alpha[num_groups][M]
    const __half* codebook, // codebook[num_codebook][256][len_vector]
    const __half* input, // input[K]
    __half* output, // output[M]
    const int M, // M
    const int K, // K
    const int group_size // group_size
);

template<bool use_bfloat16>
__global__ void _codegemm_dequant_m2v8(
  const uint32_t* __restrict__ q_weight,  // q_weight[num_codebook][K/len_vector/4][M]
  const __half* __restrict__ alpha,  // alpha[num_groups][M]
  const int4* __restrict__ codebook,
  int4* __restrict__ output,
  const int M,  // N in the new layout
  const int K,
  const int group_size  // group size for alpha
);

__global__ void _codegemm_dequant_m1v4(
    const uint32_t* __restrict__ q_weight,  // bW[num_codebook][K/len_vector/4][N]
    const __half* __restrict__ alpha,
    const __half* __restrict__ codebook,  // alpha[num_groups][M]
    __half*    __restrict__ output,
    const int M,  // N in the new layout
    const int K,  // K
    const int group_size  // group size for alpha
);

#endif

