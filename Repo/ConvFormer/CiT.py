import torch.nn as nn
import torch
from utils.layer_utils import Patchify, PositionalEncodingAppend
from utils.gen_rolls_and_masks import gen_rolls_and_masks
from utils.device_utils import get_device
from ConvFormer.CiTBlock import CiTBlock
"""Implements full Vision Transformer described in Vision Transformer paper."""
img_h = 32
img_w = 32
class CiT(nn.Module):
    def __init__(self, img_h, img_w, patch_size, d_model, mask_fidelity = 3, dropout_p = 0.01):
        super().__init__()
        internal_expand = 12
        self.conv_scale_up = nn.Sequential(
            nn.Conv2d(3,internal_expand,3,padding=1),
            Patchify(patch_size = patch_size),
        )
        self.embedding = nn.Parameter(torch.normal(0.0,0.001,(internal_expand*patch_size**2,)), requires_grad = True)
        self.positional_encoding = PositionalEncodingAppend(total_size = d_model)
        rolls, masks = gen_rolls_and_masks([(img_h, img_w)], 2, mask_fidelity = mask_fidelity, add_cls_embeddding = True)
        dropout_p = dropout_p
        block_list = [CiTBlock(num_heads = 16, mask1 = masks[0][0], mask2 = masks[0][1], rolls = rolls[0],  d_model = d_model, dropout_p=dropout_p) for i in range(8)]
        self.CiTBlocks = nn.Sequential(
            *block_list
        )
        self.fcn = nn.Sequential(
            nn.Linear(d_model, 10),
        )
    def forward(self, input):
        out0 = self.conv_scale_up(input)
        out0 = torch.cat((self.embedding.expand(out0.shape[0],1,-1), out0), dim = 1)
        out0 = self.positional_encoding(out0)
        out1 = self.CiTBlocks(out0)
        out1 = out1[:,0,:]
        out2 = self.fcn(out1)
        return out2
