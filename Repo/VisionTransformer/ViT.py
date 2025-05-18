import torch.nn as nn
import torch
from utils.layer_utils import Patchify, PositionalEncodingAppend
from utils.device_utils import get_device
from VisionTransformer.ViTBlock import ViTBlock
"""Implements full Vision Transformer described in Vision Transformer paper."""
img_h = 32
img_w = 32
class ViT(nn.Module):
    def __init__(self, img_h, img_w, patch_size, d_model, dropout_p = 0.01):
        super().__init__()
        self.conv_scale_up = nn.Sequential(
            nn.Conv2d(3,25,3,padding=1),
            Patchify(patch_size = patch_size),
        )
        self.embedding = nn.Parameter(torch.normal(0.0,0.001,(25*patch_size**2,)), requires_grad = True)
        self.positional_encoding = PositionalEncodingAppend(total_size = d_model)
        dropout_p = dropout_p
        block_list = [ViTBlock(num_heads = 16, d_model = d_model, dropout_p=dropout_p) for i in range(8)]
        self.ViTBlocks = nn.Sequential(
            *block_list
        )
        self.fcn = nn.Sequential(
            nn.Linear(d_model, 10),
        )
    def forward(self, input):
        out0 = self.conv_scale_up(input)
        out0 = torch.cat((self.embedding.expand(out0.shape[0],1,-1), out0), dim = 1)
        out0 = self.positional_encoding(out0)
        out1 = self.ViTBlocks(out0)
        out1 = out1[:,0,:]
        out2 = self.fcn(out1)
        return out2