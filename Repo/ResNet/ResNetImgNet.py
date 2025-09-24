from utils.layer_utils import Patchify
import torch.nn as nn
import math
from ResNet.ResBlock import ResBlock
"""
Not used in paper. Implemented ResNet architecture for ImageNet.
"""
class ResNet(nn.Module):
    def __init__(self, n = 9):
        super().__init__()
        self.patch_size = 1
        img_h = 16
        img_w = 16
        self.seq_len = (img_h*img_w)//self.patch_size
        self.inp_features = 3*self.patch_size
        self.transform_inp = nn.Sequential(
            Patchify(patch_size = int(math.sqrt(self.patch_size)))
        )
        self.res_block_11 = nn.Sequential(
            nn.Conv2d(self.inp_features, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
        )
        self.res_block_12 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding = 1),
            nn.BatchNorm2d(16)
        )
        self.res_block_1rest = nn.Sequential(
            *[ResBlock(16,16) for i in range(n-1)]
        )
        self.shortcut_projection1 = nn.Conv2d(16, 32, 1, stride = 2)
        self.res_block_21 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding = 1, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding = 1),
            nn.BatchNorm2d(32),
        )
        self.res_block_2rest = nn.Sequential(
            *[ResBlock(32,32) for i in range(n-1)]
        )
        self.shortcut_projection2 = nn.Conv2d(32, 64, 1, stride = 2)
        self.res_block_31 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding = 1, stride = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding = 1),
            nn.BatchNorm2d(64)
        )
        self.res_block_3rest = nn.Sequential(
            *[ResBlock(64,64) for i in range(n-1)]
        )
        self.pool = nn.AvgPool2d(4, 1)
        self.fcn = nn.Sequential(
            nn.Linear(64, 10)
        )
    def forward(self, input):
        out1 = self.transform_inp(input).reshape(-1, int(math.sqrt(self.seq_len)), int(math.sqrt(self.seq_len)), self.inp_features).transpose(-1, 1).transpose(-1,-2)
        out1 = self.res_block_11(out1)
        out2 = self.res_block_12(out1)+out1
        out3 = self.res_block_1rest(out2)+out2
        out4 = self.res_block_21(out3)+self.shortcut_projection1(out3)
        out5 = self.res_block_2rest(out4)+out4
        out6 = self.res_block_31(out5)+self.shortcut_projection2(out5)
        out7 = self.res_block_3rest(out6)+out6
        out8 = self.pool(out7)
        out9 = self.fcn(out8.reshape(out8.shape[0],-1))
        return out9