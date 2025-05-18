"""
Multi-Head Attention Layer detailed in Attention is All You Need (Vaswani, 2017) paper. Implemented to compare Transformers with the same ConvFormer architecture and Vision Transformers to ConvFormers and ResNets.
"""
import math
import torch.autograd.profiler as profiler
import gc
import torch.nn as nn
from utils.layer_utils import Patchify
import torch

class MultiHeadAttentionWReduction(nn.Module):
    def __init__(self, num_heads, d_model, patch_size = 1, reduction = False):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.patch_size = patch_size
        self.Linears = nn.ModuleList(nn.Linear(d_model, d_model) for __ in range(3))
        self.final_linear = nn.Linear(self.d_model, self.d_model)
        self.reduction = reduction
        self.patchify = Patchify(patch_size=patch_size)
    def forward(self, input):
        with profiler.record_function("KEY_QUERY_VALUE_PROJECTION"):
            keys, queries, values = (
                linear(inp).reshape(input.shape[0], -1, self.num_heads, int(self.d_model/self.num_heads)).transpose(1,2)
                for linear, inp in zip(self.Linears, (input, input, input))
            )
        with profiler.record_function("ATTENTION_MATMUL"):
            att = torch.matmul(queries,keys.transpose(-1,-2))/math.sqrt(self.d_model/self.num_heads)
            att = torch.nn.functional.softmax(att, dim=-1)
        outputs = torch.matmul(att,values).transpose(1,2).reshape(input.shape[0], -1, self.d_model)
        outputs = self.final_linear(outputs)
        if self.reduction:
            outputs = outputs.reshape(outputs.shape[0], int(math.sqrt(outputs.shape[-2])),int(math.sqrt(outputs.shape[-2])), outputs.shape[-1])
            outputs = outputs.permute(0,3,1,2)
            outputs = self.patchify(outputs)
        return outputs