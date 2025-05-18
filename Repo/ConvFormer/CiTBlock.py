from ConvFormer.LMSA_layer import MultiScaledConvHead
import torch.nn as nn
"""Implementation of one VisionTransformer block as described in the Vision Transformer paper (Dosovitsky, 2021)."""
class CiTBlock(nn.Module):
    def __init__(self, num_heads, mask1, mask2, rolls, d_model, dropout_p):
        super().__init__()
        self.BNorm1 = nn.LayerNorm(d_model)
        self.attention = MultiScaledConvHead(num_heads = num_heads, d_model = d_model, num_masks=num_heads, mask1 = mask1, mask2 = mask2, roll_matrix = rolls, reduction = False, patch_size = 1, repeat_masks = True)
        self.BNorm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.Dropout(dropout_p),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout_p)
        )
    def forward(self, input):
        out0 = self.BNorm1(input)
        out1 = self.attention(out0)+out0
        out2 = self.BNorm2(out1)
        out3 = self.MLP(out2)+out2
        return out3