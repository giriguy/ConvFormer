import torch.nn as nn
import torch
from utils.device_utils import get_device
import math
"""
Conduct patching described by paper. Spreads the pixels in each non-overlapping patch along the channel dimension.
"""
class Patchify(nn.Module):
    def __init__(self, patch_size = 2):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(patch_size, stride=patch_size)
    def forward(self, input):
        return self.unfold(input).transpose(-1,-2)
"""
Computes Batch Normalization for the ConvFormer, Vision Transformer, and the Transformer.
"""
class BatchNormTranspose(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.BNorm = nn.BatchNorm1d(channels)
    def forward(self, input):
        output = self.BNorm(input.transpose(-1,-2))
        return output.transpose(-1,-2)
"""
Appends fixed sine, and cosine positional encodings to Vision Transformer and ConvFormer.
"""
class PositionalEncodingAdd(nn.Module):
    def __init__(self, size, device = get_device()):
        super().__init__()
        self.size = size
        self.device = device
    def forward(self, input):
        len = input.shape[-2]
        positions = torch.arange(len,dtype=torch.float32).view(-1, 1).to(self.device)
        freq = 1/torch.exp(torch.linspace(0,math.ceil(math.log(len)),self.size//2, dtype=torch.float32)).view(1,-1).to(self.device)
        sin_encodings = torch.sin(torch.matmul(positions, freq)).repeat(input.shape[0],1,1)
        cos_encodings = torch.cos(torch.matmul(positions, freq)).repeat(input.shape[0],1,1)
        total_embedding = torch.cat((sin_encodings, cos_encodings), dim = -1)
        return input+total_embedding
