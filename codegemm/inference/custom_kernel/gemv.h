// Originally from https://github.com/snu-mllab/GuidedQuant/blob/main/inference/ap_gemv/gemv.h

#ifndef GEMV_CUH
#define GEMV_CUH

#include <cassert>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>

#include <torch/extension.h>
#include <cuda_runtime.h>


void codegemm_gemv(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor codebook,
    int group_size
);


torch::Tensor codegemm_dequant(
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor codebook,
    int group_size
);

#endif 
