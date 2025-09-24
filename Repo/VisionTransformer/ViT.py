import torch.nn as nn
import torch
from utils.layer_utils import Patchify, PositionalEncodingAdd
from utils.device_utils import get_device
from VisionTransformer.ViTBlock import ViTBlock
import torchvision
from torchvision.transforms import v2
import random
"""Implements full Vision Transformer described in Vision Transformer paper."""
class ViT(nn.Module):
    def __init__(self, img_h, img_w, patch_size, d_model, dropout_p = 0.01, stochastic_depth_prob = 0.1, store_att=False, num_classes = 1000, channels = 3, num_heads = 3):
        super().__init__()
        self.conv_scale_up = nn.Sequential(
            Patchify(patch_size = patch_size),
            nn.Linear(channels*patch_size**2, d_model)
        )
        self.embedding = nn.Parameter(torch.normal(0.0,0.001,(d_model,)), requires_grad = True)
        self.positional_encoding = PositionalEncodingAdd(size = d_model)
        dropout_p = dropout_p
        self.att_list = []
        block_list = [ViTBlock(num_heads = num_heads, d_model = d_model, dropout_p=dropout_p, store_att=store_att, att_store_list=self.att_list) for i in range(12)]
        self.ViTBlocks = nn.Sequential(
            *block_list
        )
        self.fcn = nn.Sequential(
            nn.Linear(d_model, num_classes),
        )
    def forward(self, input):
        out0 = self.conv_scale_up(input)
        out0 = torch.cat((self.embedding.expand(out0.shape[0],1,-1), out0), dim = 1)
        out0 = self.positional_encoding(out0)
        out1 = self.ViTBlocks(out0)
        out1 = out1[:,0,:]
        out2 = self.fcn(out1)
        return out2