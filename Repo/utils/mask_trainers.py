import torch.nn as nn
import torch
"""Module that allows for the training two different masks, and using rank approximation to mimic the filter attention map."""
class MaskTrainer(nn.Module):
    def __init__(self, seq_len = 1024, repeat_masks = True, num_masks = 16, mask_fidelity = 9, groups = 5):
        super().__init__()
        if(repeat_masks):
            self.mask1 = nn.Parameter(torch.normal(0.0,0.001,(mask_fidelity, seq_len)), requires_grad = True)
            self.mask2 = nn.Parameter(torch.normal(0.0,0.001,(mask_fidelity, seq_len)), requires_grad = True)
        else:
            self.mask1 = nn.Parameter(torch.normal(0.0,0.001,(groups, num_masks, mask_fidelity, seq_len)), requires_grad = True)
            self.mask2 = nn.Parameter(torch.normal(0.0,0.001,(groups, num_masks, mask_fidelity, seq_len)), requires_grad = True)
    def forward(self):
        att = torch.matmul(self.mask1.transpose(-1,-2), self.mask2)
        return att
model = MaskTrainer()

"""Module that allows for double rank approximation. Using two masks to approximate each mask in Rank Approximation. Those two computed masks are then used to mimic filter attention map. Not used in paper."""
class DoubleMaskTrainer(nn.Module):
    def __init__(self, img_h = 32, img_w = 32, repeat_masks = True, num_masks = 16, mask_fidelity1 = 4, mask_fidelity2 = 9):
        super().__init__()
        self.mask_fidelity1 = mask_fidelity1
        self.mask_fidelity2 = mask_fidelity2
        if(repeat_masks):
            self.mask1 = nn.Parameter(torch.normal(0.0,0.1,(mask_fidelity2,mask_fidelity1, img_h)), requires_grad = True)
            self.mask2 = nn.Parameter(torch.normal(0.0,0.1,(mask_fidelity2,mask_fidelity1, img_w)), requires_grad = True)
            self.mask3 = nn.Parameter(torch.normal(0.0,0.1,(mask_fidelity2,mask_fidelity1, img_h)), requires_grad = True)
            self.mask4 = nn.Parameter(torch.normal(0.0,0.1,(mask_fidelity2,mask_fidelity1, img_w)), requires_grad = True)
        else:
            self.mask1 = nn.Parameter(torch.normal(0.0,0.1,(num_masks, mask_fidelity2,mask_fidelity1, img_h)), requires_grad = True)
            self.mask2 = nn.Parameter(torch.normal(0.0,0.1,(num_masks,mask_fidelity2,mask_fidelity1, img_w)), requires_grad = True)
            self.mask3 = nn.Parameter(torch.normal(0.0,0.1,(num_masks,mask_fidelity2,mask_fidelity1, img_h)), requires_grad = True)
            self.mask4 = nn.Parameter(torch.normal(0.0,0.1,(num_masks,mask_fidelity2,mask_fidelity1, img_w)), requires_grad = True)
    def forward(self):
        att = torch.matmul(torch.matmul(self.mask1.transpose(-1,-2), self.mask2).reshape(self.mask_fidelity2,-1).transpose(-1,-2), torch.matmul(self.mask3.transpose(-1,-2), self.mask4).reshape(self.mask_fidelity2, -1))
        return att