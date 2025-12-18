import torch
import torch.nn as nn
import sys, os

import math
from typing import Optional

import torch
import torch.nn as nn
from codegemm.inference.plugin import *


def get_int_dtype(nbits: int) -> torch.dtype:
    if nbits <= 8:
        return torch.uint8
    if nbits <= 16:
        return torch.int16
    if nbits <= 32:
        return torch.int32
    if nbits <= 64:
        return torch.int64
    raise ValueError(f"No dtype available for {nbits}-bit codebooks")

class CodeGEMMLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        vec_len: int,
        group_size: int,
        # out_group_size: int,
        num_codebooks: int,
        nbits_per_codebook=8,
        bias=False,
        dtype=torch.half,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        assert self.in_features % vec_len == 0
        assert self.in_features % group_size == 0
        num_groups = in_features // group_size
        self.vec_len = vec_len
        self.group_size = group_size
        self.num_codebooks = num_codebooks
        self.nbits_per_codebook = nbits_per_codebook
        self.codebook_size = 2**nbits_per_codebook
        self.dtype = dtype

        self.register_buffer(
            'codes',
            torch.empty(
                (num_codebooks, in_features//vec_len//4, out_features), 
                dtype=torch.int32)
        )
        self.register_buffer(
            'codebooks',
            torch.empty(
                (num_codebooks, self.codebook_size, vec_len), 
                dtype=self.dtype)
        )
        self.register_buffer(
            'scales',
            torch.empty(
                (num_groups, out_features), 
                dtype=self.dtype)
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=self.dtype)
            )
        else:
            self.bias = None

        self.output = torch.zeros((1, 1, out_features), dtype=self.dtype, device='cuda')


    def _gemm(self, x):
        """
        x : (B, T, in_features)
        return -> (B, T, out_features)
        """
        B, T, _ = x.shape
        weight = codegemm_dequant(
            self.codes, self.scales, self.codebooks, self.group_size
        )

        x_flat = x.reshape(-1, self.in_features)
        y_flat = torch.matmul(x_flat, weight)

        return y_flat.reshape(B, T, self.out_features)

    def forward(self, x, **kwargs):
        """
        x : (B, T, in_features)
        """
        
        
        if x.numel() // self.in_features == 1:
            self.output.zero_()
            codegemm_gemv(
                x, self.output,
                self.codes, self.scales, self.codebooks, self.group_size
            )
            out = self.output
        else:
            out = self._gemm(x)

        if self.bias is not None:
            out += self.bias
        return out
