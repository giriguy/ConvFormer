import torch.nn as nn
"""
Not used in paper. One ResNet block implemented as described in the ResNet paper (He, 2015)
"""
class ResBlock(nn.Module):

    # Initialize ResBlock for Resnet Comparison
    def __init__(self, num_in, num_out):
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.stride = 1
        self.conv2d_1 = nn.Conv2d(num_in, num_out, 3, stride = self.stride, padding = 1)
        self.conv2d_2 = nn.Conv2d(num_in, num_out, 3, stride = self.stride, padding = 1)
        self.batch_norm = nn.BatchNorm2d(num_out)
        self.ReLU = nn.ReLU()

    # Compute forward pass. for ResNet layer  
    def forward(self, input):
        out0 = self.ReLU(input)
        out1 = self.conv2d_1(out0)
        out2 = self.batch_norm(out1)
        out3 = self.ReLU(out2)
        out4 = self.conv2d_2(out3)
        out5 = self.batch_norm(out4)
        return out5