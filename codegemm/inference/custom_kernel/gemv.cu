// Originally from https://github.com/snu-mllab/GuidedQuant/blob/main/inference/ap_gemv/gemv.cu

#include <cuda_fp16.h>
#include <cuda.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include <torch/extension.h>
#include "gemv.h"
#include "codegemm.h"
#include "datatype.h"

// CodeGEMM
#define K_TILE_SIZE 32
#define NUM_THREADS 256
#define M_TILE_SIZE 2048

#define THREAD_M_DEQUANT 13

void cudaError(cudaError_t errCode, const char * filename, int linenum) {
    if(errCode != cudaSuccess) {
        printf("Error : %s (%s : %d)\n", cudaGetErrorString(errCode), filename, linenum);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (cudaError(err, __FILE__, __LINE__))


////////////////////////////////////////////////////////////////////////////////
//                                     CodeGEMM
////////////////////////////////////////////////////////////////////////////////

template <int num_codebook, int len_vector>
void codegemm_gemv_templated(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor codebook,
    int group_size,
    cudaStream_t stream
) {
    uint32_t kSize = input.size(2);
    uint32_t mSize = output.size(2);

    dim3 grid((mSize + M_TILE_SIZE - 1) / M_TILE_SIZE,
              (kSize + K_TILE_SIZE - 1) / K_TILE_SIZE);
    dim3 block(NUM_THREADS);

    codegemm<num_codebook, len_vector><<<grid, block, 0, stream>>>(
        (uint32_t*) q_weight.data_ptr<int32_t>(),
        (__half*) alpha.data_ptr<at::Half>(),
        (__half*) codebook.data_ptr<at::Half>(),
        (__half*) input.data_ptr<at::Half>(),
        (__half*) output.data_ptr<at::Half>(),
        mSize, kSize, group_size
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));
}

void codegemm_gemv_stream(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor codebook,
    int group_size,
    cudaStream_t stream
) {
    TORCH_CHECK(input.scalar_type() == alpha.scalar_type() && input.scalar_type() == codebook.scalar_type() && input.scalar_type() == output.scalar_type(), "Mismatched data types between input, alpha, codebook, and output tensors.");
    // Check that input is of shape (batch_size, seq_len, input_feat)
    TORCH_CHECK(input.dim() == 3, "input tensor must be of shape (batch_size, seq_len, input_feat).");
    // Check that output is of shape (batch_size, seq_len, output_feat)
    TORCH_CHECK(output.dim() == 3, "output tensor must be of shape (batch_size, seq_len, output_feat).");

    // Only allow single batch size and sequence length
    TORCH_CHECK(input.size(0) == 1, "Batch size must be 1 for input tensor.");
    TORCH_CHECK(input.size(1) == 1, "Sequence length must be 1 for input tensor.");
    TORCH_CHECK(output.size(0) == 1, "Batch size must be 1 for output tensor.");
    TORCH_CHECK(output.size(1) == 1, "Sequence length must be 1 for output tensor.");

    // Check that input and output are both on GPU
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "input and output tensors must be on GPU.");

    // Check that all tensors are contiguous
    TORCH_CHECK(input.is_contiguous(), "input tensor must be contiguous.");
    TORCH_CHECK(output.is_contiguous(), "output tensor must be contiguous.");
    TORCH_CHECK(q_weight.is_contiguous(), "q_weight tensor must be contiguous.");
    TORCH_CHECK(alpha.is_contiguous(), "alpha tensor must be contiguous.");
    TORCH_CHECK(codebook.is_contiguous(), "q_bias tensor must be contiguous.");

    uint32_t kSize = input.size(2);
    uint32_t mSize = output.size(2);

    // Infer num_codebook and len_vector from codebook shape
    // codebook shape: [num_codebook][256][len_vector]
    TORCH_CHECK(codebook.dim() == 3, "codebook tensor must be 3-dimensional [num_codebook, 256, len_vector].");
    TORCH_CHECK(codebook.size(1) == 256, "codebook second dimension must be 256.");
    
    int num_codebook = codebook.size(0);
    int len_vector = codebook.size(2);

    // Dispatch to appropriate template instantiation based on num_codebook and len_vector
    if (num_codebook == 1 && len_vector == 4) {
        codegemm_gemv_templated<1, 4>(input, output, q_weight, alpha, codebook, group_size, stream);
    } else if (num_codebook == 2 && len_vector == 8) {
        codegemm_gemv_templated<2, 8>(input, output, q_weight, alpha, codebook, group_size, stream);
    } else {
        TORCH_CHECK(false, "Unsupported configuration: num_codebook=", num_codebook, ", len_vector=", len_vector, 
                    ". Supported configurations: (num_codebook=1, len_vector=4) and (num_codebook=2, len_vector=8).");
    }
}

void codegemm_gemv(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor codebook,
    int group_size
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    codegemm_gemv_stream(input, output, q_weight, alpha, codebook, group_size, stream);
}

torch::Tensor codegemm_dequant(
    torch::Tensor q_weight,
    torch::Tensor alpha,
    torch::Tensor codebook,
    int group_size
) {
    HANDLE_ERROR(cudaSetDevice(q_weight.device().index()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Infer num_codebook and len_vector from codebook shape
    // codebook shape: [num_codebook][256][len_vector]
    TORCH_CHECK(codebook.dim() == 3, "codebook tensor must be 3-dimensional [num_codebook, 256, len_vector].");
    TORCH_CHECK(codebook.size(1) == 256, "codebook second dimension must be 256.");
    
    int num_codebook = codebook.size(0);
    int len_vector = codebook.size(2);

    // q_weight shape: [num_codebook][K/len_vector/4][M]
    const int mSize = q_weight.size(2);
    const int kSize = q_weight.size(1) * len_vector * 4;

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(q_weight.device());

    dim3 grid((mSize+THREAD_M_DEQUANT-1)/THREAD_M_DEQUANT);
    dim3 block(32*THREAD_M_DEQUANT);
    
    at::Tensor weight_transposed;  // Kernels output [M][K], we need [K][M]
    
    if (num_codebook == 1 && len_vector == 4) {
        // int shared_mem_size = 256 * 2 * sizeof(__half2);  // Codebook in shared memory
        int shared_mem_size = 16 * (1 * 256 * 4 + 32 * 9);
        
        weight_transposed = torch::empty({mSize, kSize}, options);
        
        _codegemm_dequant_m1v4<<<grid, block, shared_mem_size, stream>>>(
            (uint32_t*) q_weight.data_ptr<int32_t>(),
            (__half*) alpha.data_ptr<at::Half>(),
            (__half*) codebook.data_ptr<at::Half>(),
            (__half*) weight_transposed.data_ptr<at::Half>(),
            mSize, kSize, group_size
        );
    } else if (num_codebook == 2 && len_vector == 8) {
        
        // int shared_mem_size = 2 * 256 * 8 * sizeof(int4);  // Two codebooks in shared memory
        int shared_mem_size = 16 * (2 * 256 * 8 + 32 * 9);
        cudaFuncSetAttribute(
            _codegemm_dequant_m2v8<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size
        );
        weight_transposed = torch::empty({mSize, kSize}, options);
        // use_bfloat16 = false
        _codegemm_dequant_m2v8<false><<<grid, block, shared_mem_size, stream>>>(
            (uint32_t*) q_weight.data_ptr<int32_t>(),
            (__half*) alpha.data_ptr<at::Half>(),
            (int4*) codebook.data_ptr<at::Half>(),
            (int4*) weight_transposed.data_ptr<at::Half>(),
            mSize, kSize, group_size
        );
    } else {
        TORCH_CHECK(false, "Unsupported configuration: num_codebook=", num_codebook, ", len_vector=", len_vector, 
                    ". Supported configurations: (num_codebook=1, len_vector=4) and (num_codebook=2, len_vector=8).");
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err));

    // Transpose to get [K][M] layout
    return weight_transposed.t();
}