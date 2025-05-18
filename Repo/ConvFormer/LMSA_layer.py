"""This is the implementation of the Learned Masked Self Attention Layer described in the paper. It is implemented according to the algorithm detailed in the paper."""
import math
import torch.autograd.profiler as profiler
import gc
import torch.nn as nn
import torch
from utils.layer_utils import Patchify
class MultiScaledConvHead(nn.Module):
    def __init__(self, num_heads, d_model, mask1, mask2, num_masks, roll_matrix, reduction = False, patch_size = 2, repeat_masks = True, include_fcn = False):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_masks = num_masks
        self.Linears = nn.ModuleList(nn.Linear(d_model, d_model) for __ in range(3))
        if(repeat_masks):
            self.mask1 = nn.Parameter(mask1.clone().repeat(num_masks, 1, 1), requires_grad = True)
            self.mask2 = nn.Parameter(mask2.clone().repeat(num_masks, 1, 1), requires_grad = True)
        else:
            self.mask1 = nn.Parameter(mask1.clone(), requires_grad = True)
            self.mask2 = nn.Parameter(mask2.clone(), requires_grad = True)
        self.register_buffer(name='roll_back', tensor=roll_matrix.clone())
        self.roll_back = self.roll_back
        self.reduction = reduction
        self.patch_size = patch_size
        self.patchify = Patchify(patch_size=2)
        self.epsilon = 0.00000001
        self.final_linear = nn.Linear(d_model, d_model)
        self.include_fcn = include_fcn
    def forward(self, input):
        keys, queries, values = (
            linear(inp).reshape(input.shape[0], -1, self.num_heads, int(self.d_model/self.num_heads)).transpose(1,2)
            for linear, inp in zip(self.Linears, (input, input, input))
        )
        learnable_mask = torch.matmul(self.mask1.transpose(-1,-2), self.mask2)
        # print(learnable_mask.shape)
        att = torch.matmul(queries,keys.transpose(-1,-2))/math.sqrt(self.d_model/self.num_heads)
        att = att.reshape(input.shape[0], int(self.num_heads/self.num_masks),self.num_masks, att.shape[-2],att.shape[-1])
        del keys
        del queries
        att = att*torch.gather(learnable_mask,-1,self.roll_back.expand_as(learnable_mask)).expand_as(att)
        # print(att.shape)
        del learnable_mask
        att = att.reshape(input.shape[0], self.num_heads, att.shape[-2], att.shape[-1])
        #Final attention normed to prevent exploding and vanishing gradients.
        att = att/(torch.norm(att, dim = (-1,-2))[..., None,None]+self.epsilon)
        outputs = torch.matmul(att,values).transpose(1,2).reshape(input.shape[0], -1, self.d_model)
        outputs = self.final_linear(outputs)
        if self.reduction:
            outputs = outputs.reshape(outputs.shape[0], int(math.sqrt(outputs.shape[-2])),int(math.sqrt(outputs.shape[-2])), outputs.shape[-1])
            outputs = outputs.permute(0,3,1,2)
            outputs = self.patchify(outputs)
        if self.include_fcn:
            outputs = self.final_linear(outputs)
        del att
        del values
        return outputs
gc.collect()