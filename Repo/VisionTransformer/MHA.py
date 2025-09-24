import torch
import torch.nn as nn
import math
import torch.autograd.profiler as profiler
import gc
"""
Multi-Head Attention Layer detailed in Attention is All You Need (Vaswani, 2017) paper. Implemented to compare Transformers with the same ConvFormer architecture and Vision Transformers to ConvFormers and ResNets.
"""

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, store_att = False, att_store_list = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.Linears = nn.ModuleList(nn.Linear(d_model, d_model) for __ in range(3))
        self.final_linear = nn.Linear(d_model, d_model)
        self.store_att = store_att
        self.att_store_list = att_store_list
    def forward(self, input):
        with profiler.record_function("KEY_QUERY_VALUE_PROJECTION"):
            keys, queries, values = (
                linear(inp).reshape(input.shape[0], -1, self.num_heads, int(self.d_model/self.num_heads)).transpose(1,2)
                for linear, inp in zip(self.Linears, (input, input, input))
            )
        with profiler.record_function("ATTENTION_MATMUL"):
            att = torch.matmul(queries,keys.transpose(-1,-2))/math.sqrt(self.d_model/self.num_heads)
            att = torch.nn.functional.softmax(att, dim=-1)
        if self.store_att:
            self.att_store_list.append(att)
        outputs = torch.matmul(att,values).transpose(1,2).reshape(input.shape[0], -1, self.d_model)
        outputs = self.final_linear(outputs)
        return outputs