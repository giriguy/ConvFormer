# Import all the necessary libraries
import torch.nn as nn
import torch
from utils.layer_utils import Patchify, PositionalEncodingAdd
from utils.gen_rolls_and_masks import gen_rolls_and_masks
from utils.device_utils import get_device
from IBiT.IBiTBlock import IBiTBlock

"""Implements full IBiT described in the paper."""

class IBiT(nn.Module):
    # Initialize IBiT
    def __init__(self, img_h, img_w, patch_size, d_model, mask_fidelity = 3, dropout_p = 0.01, should_train = True, store_att=False, channels = 3, num_classes = 1000, num_heads = 3):
        super().__init__()

        # Define linear embedding layer
        self.scale_up = nn.Sequential(
            Patchify(patch_size = patch_size),
            nn.Linear(channels*patch_size**2, d_model)
        )

        # Instantialize other layers including IBiT Blocks
        self.embedding = nn.Parameter(torch.normal(0.0,0.001,(d_model,)), requires_grad = True)
        self.positional_encoding = PositionalEncodingAdd(size = d_model)
        rolls, masks = gen_rolls_and_masks([(img_h, img_w)], patch_size = patch_size, mask_fidelity = mask_fidelity, add_cls_embeddding = True, should_train=should_train, filt_type = 'radial')
        dropout_p = dropout_p
        self.att_list = []
        block_list = [IBiTBlock(num_heads = num_heads, mask1 = masks[0][0], mask2 = masks[0][1], rolls = rolls[0],  d_model = d_model, dropout_p=dropout_p, store_att=store_att, att_store_list=self.att_list) for i in range(12)]
        self.CiTBlocks = nn.Sequential(
            *block_list
        )

        # Instantialize final fully connected layer
        self.fcn = nn.Sequential(
            nn.Linear(d_model, num_classes),
        )

    # Compute forward pass for IBiT
    def forward(self, input):
        out0 = self.scale_up(input)
        out0 = torch.cat((self.embedding.expand(out0.shape[0],1,-1), out0), dim = 1)
        out0 = self.positional_encoding(out0)
        out1 = self.CiTBlocks(out0)
        out1 = out1[:,0,:]
        out2 = self.fcn(out1)
        return out2
