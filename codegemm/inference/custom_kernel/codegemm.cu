/*
CodeGEMM
Copyright (c) 2025-present NAVER Cloud Corp.
Apache-2.0
*/

#include <cuda_fp16.h>
#include <stdio.h>
#include <cstdio>
#include <ctime>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>
#include "codegemm.h"
#include "typetraits.h"
#include "datatype.h"

#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <assert.h>

#define K_TILE_SIZE 32
#define NUM_THREADS 256
#define M_TILE_SIZE 2048

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
) {
    __shared__ __half lut[K_TILE_SIZE/len_vector][num_codebook][256];
    __half base[num_codebook] = {0.0f};
    const __half* _codebook[num_codebook];
    
    for (int i = 0; i < num_codebook; i++) {
        _codebook[i] = &codebook[threadIdx.x * len_vector + 256 * len_vector * i];
    }
    
    for (int i = threadIdx.y; i < K_TILE_SIZE / len_vector; i++) {
        const __half* _inp = &input[blockIdx.y * K_TILE_SIZE + i * len_vector];
    
        for (int k = 0; k < num_codebook; k++) {
        for (int j = 0; j < len_vector; j++) {
            base[k] += (_codebook[k][j]) * _inp[j];
        }
        lut[i][k][threadIdx.x] = base[k];
        base[k] = 0.0f; // Reset for next iteration
        }
    }
    __syncthreads();
    
    // Compute output in parallel
    int m_start = blockIdx.x * M_TILE_SIZE + threadIdx.x * 2;
    int m_end = min((blockIdx.x + 1) * M_TILE_SIZE, M);
    int m_step = blockDim.x * 2;
    
    const uint32_t* base_addr = &q_weight[blockIdx.y * K_TILE_SIZE / 32 * 8 / len_vector * M];
    int group_idx = (blockIdx.y * K_TILE_SIZE) / group_size;
    
    for (int m = m_start; m < m_end; m += m_step) {
        __half reg_o0 = 0.0f;
        __half reg_o1 = 0.0f;
    
        __half reg_a0 = (alpha[group_idx * M + m]);
        __half reg_a1 = (alpha[group_idx * M + m + 1]);
    
        for (int i = 0; i < K_TILE_SIZE/len_vector; i++) {
          const uint32_t* temp_base_addr = base_addr + m + M * (i/len_vector);
            for (int k = 0; k < num_codebook; k++) {
                uint32_t reg = *(temp_base_addr + M * K / 32 * 8 / len_vector * k);
                reg_o0 += lut[i][k][(reg >> (8 * (i % 4))) & 255];
    
                reg = *(temp_base_addr + M * K / 32 * 8 / len_vector * k + 1);
                reg_o1 += lut[i][k][(reg >> (8 * (i % 4))) & 255];
            }
        }
    
        reg_o0 *= reg_a0;
        reg_o1 *= reg_a1;
    
        atomicAdd((half2*)&output[m], __halves2half2((reg_o0), (reg_o1)));
    }
}


template<bool use_bfloat16>
__global__ void _codegemm_dequant_m2v8(
  const uint32_t* __restrict__ q_weight,  // q_weight[num_codebook][K/len_vector/4][M]
  const __half* __restrict__ alpha,  // alpha[num_groups][M]
  const int4* __restrict__ codebook,
  int4* __restrict__ output,
  const int M,  // N in the new layout
  const int K,
  const int group_size  // group size for alpha
) {
  // M = N (output rows), K is unchanged
  int row_id = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  bool pred = row_id < M;
  int lane = threadIdx.x % 8;

  int c_gl_stride = K / 8;
  int c_gl_wr = c_gl_stride * row_id + (threadIdx.x % 32) * 8;

  extern __shared__ int4 sh[];
  int4* sh_code = sh;
  int4* sh_code0 = sh_code;
  int4* sh_code1 = sh_code + 256 * 8;

  // Load codebook to shared memory
  for (int i = threadIdx.x; i < 2 * 256; i += blockDim.x) {
    int4 dec = codebook[i];
    #pragma unroll
    for (int j = 0; j < 8; j++)
      sh_code[8 * i + (j + lane) % 8] = dec;
  }
  __syncthreads();

  if (!pred) return;

  // New layout: bW[num_codebook][K/len_vector/4][N]
  // num_codebook = 2, len_vector = 8
  // K/len_vector/4 = K/32
  int k_packed_size = K / 32;  // K/8/4
  
  for (int k_idx = threadIdx.x % 32; k_idx < k_packed_size; k_idx += 32) {
    // Read packed data for both codebooks
    // q_weight layout: [codebook=2][k_idx][row_id]
    uint32_t packed0 = q_weight[0 * k_packed_size * M + k_idx * M + row_id];
    uint32_t packed1 = q_weight[1 * k_packed_size * M + k_idx * M + row_id];
    
    // Each uint32_t contains 4 uint8_t codes
    #pragma unroll
    for (int byte_idx = 0; byte_idx < 4; byte_idx++) {
      uint8_t code0 = (packed0 >> (byte_idx * 8)) & 0xFF;
      uint8_t code1 = (packed1 >> (byte_idx * 8)) & 0xFF;
      
      // Calculate k position for this element
      // k_idx * 4 + byte_idx gives the position in terms of 8-element vectors
      // Multiply by 8 to get the actual k position
      int k_pos = (k_idx * 4 + byte_idx) * 8;
      int group_idx = k_pos / group_size;
      __half alpha_val = alpha[group_idx * M + row_id];
      
      int4 chunk;
      if constexpr (use_bfloat16) {
        #if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
        nv_bfloat162* a0 = reinterpret_cast<nv_bfloat162*>(&sh_code0[8 * code0 + lane]);
        nv_bfloat162* a1 = reinterpret_cast<nv_bfloat162*>(&sh_code1[8 * code1 + lane]);
        nv_bfloat16 alpha_bf16 = __float2bfloat16(__half2float(alpha_val));
        nv_bfloat162 alpha2 = __bfloat162bfloat162(alpha_bf16);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          nv_bfloat162 sum = __hadd2(a0[j], a1[j]);
          reinterpret_cast<nv_bfloat162*>(&chunk)[j] = __hmul2(sum, alpha2);
        }
        #endif
      } else {
        half2* a0 = reinterpret_cast<half2*>(&sh_code0[8 * code0 + lane]);
        half2* a1 = reinterpret_cast<half2*>(&sh_code1[8 * code1 + lane]);
        half2 alpha2 = __half2half2(alpha_val);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          half2 sum = __hadd2(a0[j], a1[j]);
          reinterpret_cast<half2*>(&chunk)[j] = __hmul2(sum, alpha2);
        }
      }
      
      // Output index: row_id * (K/8) + k_idx * 4 + byte_idx
      int out_idx = row_id * (K / 8) + k_idx * 4 + byte_idx;
      output[out_idx] = chunk;
    }
  }
}


__global__ void _codegemm_dequant_m1v4(
  const uint32_t* __restrict__ q_weight,  // bW[num_codebook][K/len_vector/4][N]
  const __half* __restrict__ alpha,
  const __half* __restrict__ codebook,  // alpha[num_groups][M]
  __half*    __restrict__ output,
  const int M,  // N in the new layout
  const int K,  // K
  const int group_size  // group size for alpha
) {

  extern __shared__ __half2 sh_cb[];

  // Load codebook to shared memory using vectorized loads
  // Codebook: [256 entries][4 halfs] = [256 entries][2 half2]
  const float2* codebook2 = reinterpret_cast<const float2*>(codebook);
  float2* sh_cb2 = reinterpret_cast<float2*>(sh_cb);
  int total_vec2 = 256 * 2 / 2;  // 256 entries * 2 half2 per entry / 2 half2 per float2
  for (int idx = threadIdx.x; idx < total_vec2; idx += blockDim.x) {
      sh_cb2[idx] = codebook2[idx];
  }
  __syncthreads();

  const int WARP = 32;
  int warp_id = (blockIdx.x * (blockDim.x / WARP)) + (threadIdx.x / WARP);
  if (warp_id >= M) return;

  int lane = threadIdx.x & (WARP - 1);
  
  // New layout: bW[num_codebook=1][K/len_vector/4][N]
  // num_codebook = 1, len_vector = 4
  // K/len_vector/4 = K/16
  int k_packed_size = K / 16;  // K/4/4
  
  // output: [M × K] __half
  __half* row_out = output + warp_id * K;
  
  // Reorganize loop for better memory coalescing
  // Process in chunks where warp threads write to consecutive memory
  int iterations = (k_packed_size + WARP - 1) / WARP;
  
  for (int iter = 0; iter < iterations; iter++) {
      int k_idx = iter * WARP + lane;
      
      if (k_idx < k_packed_size) {
          // Read packed data: q_weight[codebook=0][k_idx][warp_id]
          uint32_t packed = q_weight[k_idx * M + warp_id];
          
          // Each uint32_t contains 4 uint8_t codes
          // Extract all 4 codes
          uint8_t c0 = (packed >>  0) & 0xFF;
          uint8_t c1 = (packed >>  8) & 0xFF;
          uint8_t c2 = (packed >> 16) & 0xFF;
          uint8_t c3 = (packed >> 24) & 0xFF;
          
          // Fetch from shared memory (codebook lookup)
          // Each code maps to 2 half2 (4 halfs) = 1 float2
          const float2* sh_cb2_ptr = reinterpret_cast<const float2*>(sh_cb);
          float2 decoded0 = sh_cb2_ptr[c0];
          float2 decoded1 = sh_cb2_ptr[c1];
          float2 decoded2 = sh_cb2_ptr[c2];
          float2 decoded3 = sh_cb2_ptr[c3];
          
          // Apply alpha scaling
          // Each k_idx produces 4 codes * 4 halfs = 16 halfs
          // Calculate k position and corresponding alpha values
          int k_pos_base = k_idx * 16;  // Starting k position for this iteration
          __half2* decoded0_h2 = reinterpret_cast<__half2*>(&decoded0);
          __half2* decoded1_h2 = reinterpret_cast<__half2*>(&decoded1);
          __half2* decoded2_h2 = reinterpret_cast<__half2*>(&decoded2);
          __half2* decoded3_h2 = reinterpret_cast<__half2*>(&decoded3);
          
          // Apply alpha for each 4-element group
          #pragma unroll
          for (int i = 0; i < 2; i++) {
              int k_pos = k_pos_base + i * 2;
              int group_idx = k_pos / group_size;
              __half alpha_val = alpha[group_idx * M + warp_id];
              half2 alpha2 = __half2half2(alpha_val);
              decoded0_h2[i] = __hmul2(decoded0_h2[i], alpha2);
          }
          
          #pragma unroll
          for (int i = 0; i < 2; i++) {
              int k_pos = k_pos_base + 4 + i * 2;
              int group_idx = k_pos / group_size;
              __half alpha_val = alpha[group_idx * M + warp_id];
              half2 alpha2 = __half2half2(alpha_val);
              decoded1_h2[i] = __hmul2(decoded1_h2[i], alpha2);
          }
          
          #pragma unroll
          for (int i = 0; i < 2; i++) {
              int k_pos = k_pos_base + 8 + i * 2;
              int group_idx = k_pos / group_size;
              __half alpha_val = alpha[group_idx * M + warp_id];
              half2 alpha2 = __half2half2(alpha_val);
              decoded2_h2[i] = __hmul2(decoded2_h2[i], alpha2);
          }
          
          #pragma unroll
          for (int i = 0; i < 2; i++) {
              int k_pos = k_pos_base + 12 + i * 2;
              int group_idx = k_pos / group_size;
              __half alpha_val = alpha[group_idx * M + warp_id];
              half2 alpha2 = __half2half2(alpha_val);
              decoded3_h2[i] = __hmul2(decoded3_h2[i], alpha2);
          }
          
          // Write to output using vectorized stores
          float2* row_out2 = reinterpret_cast<float2*>(row_out + k_idx * 16);
          row_out2[0] = decoded0;
          row_out2[1] = decoded1;
          row_out2[2] = decoded2;
          row_out2[3] = decoded3;
      }
  }
}

// Explicit template instantiations for codegemm kernel
template __global__ void codegemm<1, 4>(
    const uint32_t* q_weight,
    const __half* alpha,
    const __half* codebook,
    const __half* input,
    __half* output,
    const int M,
    const int K,
    const int group_size
);

template __global__ void codegemm<2, 8>(
    const uint32_t* q_weight,
    const __half* alpha,
    const __half* codebook,
    const __half* input,
    __half* output,
    const int M,
    const int K,
    const int group_size
);

// Explicit template instantiation for _codegemm_dequant_m2v8 kernel
template __global__ void _codegemm_dequant_m2v8<false>(
    const uint32_t* __restrict__ q_weight,
    const __half* __restrict__ alpha,
    const int4* __restrict__ codebook,
    int4* __restrict__ output,
    const int M,
    const int K,
    const int group_size
);
