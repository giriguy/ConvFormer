import torch
import torch.nn as nn
from utils.device_utils import get_device
from utils.layer_utils import Patchify, BatchNormTranspose, PositionalEncodingAppend
from TransformerResNet.MHA_w_reduction import MultiHeadAttentionWReduction

"""Implements Transformer with the same model architecture as ResNets and ConvFormers to show that architectural differences are not responsible for the difference in performance between Transformers and ConvFormers."""
patch_size = 4
class TransformerResNet(nn.Module):
    def __init__(self, img_h, img_w, d_model, patch_size, init_heads = 16, dropout_p = 0.01):
        super().__init__()
        res_block1_heads = init_heads
        res_block2_heads = res_block1_heads*2
        res_block3_heads = res_block2_heads*2
        #Should be square number
        patch_size = patch_size
        d_model1 = d_model
        d_model2 = d_model1*(patch_size**2)
        d_model3 = d_model2*(patch_size**2)
        dropout_p = dropout_p
        self.conv_scale_up = nn.Sequential(
            nn.Conv2d(3,4,3,padding=1),
            Patchify(patch_size = patch_size),
            PositionalEncodingAppend(d_model1, device = get_device())
        )
        self.res_block_11 = nn.Sequential(
            MultiHeadAttentionWReduction(num_heads = res_block1_heads, d_model = d_model1, reduction = False),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.res_block_12 = nn.Sequential(
            nn.ReLU(),
            MultiHeadAttentionWReduction(num_heads = res_block1_heads, d_model = d_model1, reduction = False),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiHeadAttentionWReduction(num_heads = res_block1_heads, d_model = d_model1, reduction = False),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.res_block_13 = nn.Sequential(
            nn.ReLU(),
            MultiHeadAttentionWReduction(num_heads = res_block1_heads, d_model = d_model1, reduction = False),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiHeadAttentionWReduction(num_heads = res_block1_heads, d_model = d_model1, reduction = False),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.res_block_14 = nn.Sequential(
            nn.ReLU(),
            MultiHeadAttentionWReduction(num_heads = res_block1_heads, d_model = d_model1, reduction = False),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiHeadAttentionWReduction(num_heads = res_block1_heads, d_model = d_model1, reduction = False),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.shortcut_projection1 = MultiHeadAttentionWReduction(num_heads = res_block1_heads, d_model = d_model1, reduction = True, patch_size = 2)
        self.res_block_21 = nn.Sequential(
            nn.ReLU(),
            MultiHeadAttentionWReduction(num_heads = res_block1_heads, d_model = d_model1, reduction = True, patch_size = 2),
            BatchNormTranspose(d_model2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiHeadAttentionWReduction(num_heads = res_block2_heads, d_model = d_model2, reduction = False),
            BatchNormTranspose(d_model2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.res_block_22 = nn.Sequential(
            nn.ReLU(),
            MultiHeadAttentionWReduction(num_heads = res_block2_heads, d_model = d_model2, reduction = False),
            BatchNormTranspose(d_model2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiHeadAttentionWReduction(num_heads = res_block2_heads, d_model = d_model2, reduction = False),
            BatchNormTranspose(d_model2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.res_block_23 = nn.Sequential(
            nn.ReLU(),
            MultiHeadAttentionWReduction(num_heads = res_block2_heads, d_model = d_model2, reduction = False),
            BatchNormTranspose(d_model2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiHeadAttentionWReduction(num_heads = res_block2_heads, d_model = d_model2, reduction = False),
            BatchNormTranspose(d_model2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.shortcut_projection2 = MultiHeadAttentionWReduction(num_heads = res_block2_heads, d_model = d_model2, reduction = True, patch_size = 2)
        self.res_block_31 = nn.Sequential(
            nn.ReLU(),
            MultiHeadAttentionWReduction(num_heads = res_block2_heads, d_model = d_model2, reduction = True, patch_size = 2),
            BatchNormTranspose(d_model3),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiHeadAttentionWReduction(num_heads = res_block3_heads, d_model = d_model3, reduction = False),
            BatchNormTranspose(d_model3),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.res_block_32 = nn.Sequential(
            nn.ReLU(),
            MultiHeadAttentionWReduction(num_heads = res_block3_heads, d_model = d_model3, reduction = False),
            BatchNormTranspose(d_model3),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiHeadAttentionWReduction(num_heads = res_block3_heads, d_model = d_model3, reduction = False),
            BatchNormTranspose(d_model3),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.res_block_33 = nn.Sequential(
            nn.ReLU(),
            MultiHeadAttentionWReduction(num_heads = res_block3_heads, d_model = d_model3, reduction = False),
            BatchNormTranspose(d_model3),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiHeadAttentionWReduction(num_heads = res_block3_heads, d_model = d_model3, reduction = False),
            BatchNormTranspose(d_model3),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.fcn = nn.Sequential(
            nn.Linear(d_model3, 10)
        )
        self.LogSoftmax = nn.LogSoftmax(dim=1)
    def forward(self, input):
        out0 = self.conv_scale_up(input)
        out1 = self.res_block_11(out0)+out0
        out2 = self.res_block_12(out1)+out1
        out3 = self.res_block_13(out2)+out2
        out4 = self.res_block_14(out3)+out3
        out5 = self.res_block_21(out4)+self.shortcut_projection1(out4)
        out6 = self.res_block_22(out5)+out5
        out7 = self.res_block_23(out6)+out6
        out8 = self.res_block_31(out7)+self.shortcut_projection2(out7)
        out9 = self.res_block_32(out8)+out8
        out10 = self.res_block_33(out9)+out9
        out10 = torch.mean(out10, dim=-2)
        out11 = self.fcn(out10)
        out11 = out11 - torch.amax(out11, 1, keepdim=True)
        out11 = self.LogSoftmax(out11)
        return out11
"""
Implements Reduced-Size Transformer with the same model architecture as ResNets and ConvFormers.
"""
patch_size = 2
class TransformerResNetSmall(nn.Module):
    def __init__(self, img_h, img_w, d_model, patch_size, init_heads = 16, dropout_p = 0.01):
        super().__init__()
        res_block1_heads = init_heads
        res_block2_heads = res_block1_heads*2
        res_block3_heads = res_block2_heads*2
        #Should be square number
        patch_size = patch_size
        d_model1 = d_model
        d_model2 = d_model1*(patch_size**2)
        d_model3 = d_model2*(patch_size**2)
        dropout_p = dropout_p
        self.conv_scale_up = nn.Sequential(
            nn.Conv2d(3,4,3,padding=1),
            Patchify(patch_size = patch_size),
            PositionalEncodingAppend(d_model1, device = get_device())
        )
        self.res_block_11 = nn.Sequential(
            MultiHeadAttentionWReduction(num_heads = res_block1_heads, d_model = d_model1, reduction = False),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.res_block_12 = nn.Sequential(
            nn.ReLU(),
            MultiHeadAttentionWReduction(num_heads = res_block1_heads, d_model = d_model1, reduction = False),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiHeadAttentionWReduction(num_heads = res_block1_heads, d_model = d_model1, reduction = False),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.shortcut_projection1 = MultiHeadAttentionWReduction(num_heads = res_block1_heads, d_model = d_model1, reduction = True, patch_size=patch_size)
        self.res_block_21 = nn.Sequential(
            nn.ReLU(),
            MultiHeadAttentionWReduction(num_heads = res_block1_heads, d_model = d_model1, reduction = True, patch_size=patch_size),
            BatchNormTranspose(d_model2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiHeadAttentionWReduction(num_heads = res_block2_heads, d_model = d_model2, reduction = False),
            BatchNormTranspose(d_model2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.shortcut_projection2 = MultiHeadAttentionWReduction(num_heads = res_block2_heads, d_model = d_model2, reduction = True, patch_size=patch_size)
        self.res_block_31 = nn.Sequential(
            nn.ReLU(),
            MultiHeadAttentionWReduction(num_heads = res_block2_heads, d_model = d_model2, reduction = True, patch_size=patch_size),
            BatchNormTranspose(d_model3),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiHeadAttentionWReduction(num_heads = res_block3_heads, d_model = d_model3, reduction = False),
            BatchNormTranspose(d_model3),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.fcn = nn.Sequential(
            nn.Linear(d_model3, 10)
        )
        self.LogSoftmax = nn.LogSoftmax(dim=1)
    def forward(self, input):
        out0 = self.conv_scale_up(input)

        out1 = self.res_block_11(out0)+out0

        out2 = self.res_block_12(out1)+out1

        out3 = self.res_block_21(out2)+self.shortcut_projection1(out2)

        out4 = self.res_block_31(out3)+self.shortcut_projection2(out3)

        out5 = torch.mean(out4, dim=-2)

        out6 = self.fcn(out5)

        out6 = out6 - torch.amax(out6, 1, keepdim=True)
        out6 = self.LogSoftmax(out6)
        return out6   
