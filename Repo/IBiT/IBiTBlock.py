# Import all the necessary libraries
from ConvFormer.LMSA_layer import MultiScaledConvHead
from VisionTransformer.MHA import MultiHeadAttention
import torch.nn as nn
from torchvision.ops import StochasticDepth

"""Implementation of one IBiT block as described in the paper"""
class IBiTBlock(nn.Module):

    # Initialize single IBiT Block
    def __init__(self, num_heads, mask1, mask2, rolls, d_model, dropout_p, store_att=False, att_store_list=None, stochastic_depth_prob=0.1):
        super().__init__()

        # Instantialize LMSALayer, batch norm layers
        self.BNorm1 = nn.LayerNorm(d_model)
        self.attention = MultiScaledConvHead(num_heads = num_heads, d_model = d_model, num_masks=num_heads, mask1 = mask1, mask2 = mask2, roll_matrix = rolls, reduction = False, patch_size = 1, repeat_masks = True, include_fcn=True, store_att=store_att, att_store_list=att_store_list)
        self.BNorm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.Dropout(dropout_p),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout_p)
        )

        # Initialize stochastic depth layer
        self.stochastic_depth = StochasticDepth(p = stochastic_depth_prob, mode = 'row')
        self.mode = 'train'

    # Compute forward pass for IBiT    
    def forward(self, input):
        out0 = self.BNorm1(input)
        out1 = self.stochastic_depth(self.attention(out0))+out0
        out2 = self.BNorm2(out1)
        out3 = self.MLP(out2)+out2
        return out3