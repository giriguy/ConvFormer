from VisionTransformer.MHA import MultiHeadAttention
import torch.nn as nn
from torchvision.ops import StochasticDepth
"""Implementation of one VisionTransformer block as described in the Vision Transformer paper (Dosovitsky, 2021)."""
class ViTBlock(nn.Module):
    def __init__(self, num_heads, d_model, dropout_p, store_att=False, att_store_list=None, stochastic_depth_prob = 0.1):
        super().__init__()
        self.BNorm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(num_heads = num_heads, d_model = d_model, store_att=store_att, att_store_list=att_store_list)
        self.BNorm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.Dropout(dropout_p),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout_p)
        )
        self.stochastic_depth = StochasticDepth(p = stochastic_depth_prob, mode = 'row')
    def forward(self, input):
        out0 = self.BNorm1(input)
        out1 = self.stochastic_depth(self.attention(out0))+out0
        out2 = self.BNorm2(out1)
        out3 = self.MLP(out2)+out2
        return out3