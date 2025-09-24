"Not used in paper. First version of model trained on CIFAR-10"
import torch.nn as nn
import torch
from utils.layer_utils import Patchify, PositionalEncodingAppend, BatchNormTranspose
from IBiT.LMSA_layer import MultiScaledConvHead
from utils.device_utils import get_device
from utils.gen_rolls_and_masks import gen_rolls_and_masks
class ConvFormerResNet(nn.Module):
    def __init__(self, img_h, img_w, patch_size, img_sizes = [(32,32), (16,16), (8,8)], mask_fidelity = 1):
        super().__init__()
        rolls, masks = gen_rolls_and_masks(img_sizes=img_sizes, patch_size = patch_size, filt_type = "filter", mask_fidelity = mask_fidelity, should_train = True)
        
        res_block1_heads = 16
        res_block2_heads = 32
        res_block3_heads = 64
        res_block1_masks = 16
        res_block2_masks = 32
        res_block3_masks = 64
        #Should be square number
        self.patch_size = patch_size
        self.reduce_patch_size = 2
        d_model1 = 32
        d_model2 = d_model1*(self.reduce_patch_size**2)
        d_model3 = d_model2*(self.reduce_patch_size**2)
        dropout_p = 0.01
        self.img_h = img_h
        self.img_w = img_w
        self.conv_scale_up = nn.Sequential(
            nn.Conv2d(3,4,3,padding=1),
            Patchify(patch_size = self.patch_size),
            PositionalEncodingAppend(d_model1, device = get_device())
        )
        self.res_block_11 = nn.Sequential(
            MultiScaledConvHead(num_heads = res_block1_heads, d_model = d_model1, num_masks = res_block1_masks, reduction = False, mask1 = masks[0][0], mask2 = masks[0][1], roll_matrix = rolls[0]),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.res_block_12 = nn.Sequential(
            nn.ReLU(),
            MultiScaledConvHead(num_heads = res_block1_heads, d_model = d_model1, num_masks = res_block1_masks, reduction = False, mask1 = masks[0][0], mask2 = masks[0][1], roll_matrix = rolls[0]),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiScaledConvHead(num_heads = res_block1_heads, d_model = d_model1, num_masks = res_block1_masks, reduction = False, mask1 = masks[0][0], mask2 = masks[0][1], roll_matrix = rolls[0]),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.res_block_13 = nn.Sequential(
            nn.ReLU(),
            MultiScaledConvHead(num_heads = res_block1_heads, d_model = d_model1, num_masks = res_block1_masks, reduction = False, mask1 = masks[0][0], mask2 = masks[0][1], roll_matrix = rolls[0]),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiScaledConvHead(num_heads = res_block1_heads, d_model = d_model1, num_masks = res_block1_masks, reduction = False, mask1 = masks[0][0], mask2 = masks[0][1], roll_matrix = rolls[0]),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.res_block_14 = nn.Sequential(
            nn.ReLU(),
            MultiScaledConvHead(num_heads = res_block1_heads, d_model = d_model1, num_masks = res_block1_masks, reduction = False, mask1 = masks[0][0], mask2 = masks[0][1], roll_matrix = rolls[0]),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiScaledConvHead(num_heads = res_block1_heads, d_model = d_model1, num_masks = res_block1_masks, reduction = False, mask1 = masks[0][0], mask2 = masks[0][1], roll_matrix = rolls[0]),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.shortcut_projection1 = MultiScaledConvHead(num_heads = res_block1_heads, d_model = d_model1, num_masks = res_block1_masks, reduction = True, patch_size = self.reduce_patch_size, mask1 = masks[0][0], mask2 = masks[0][1], roll_matrix = rolls[0])
        self.res_block_21 = nn.Sequential(
            nn.ReLU(),
            MultiScaledConvHead(num_heads = res_block1_heads, d_model = d_model1, num_masks = res_block1_masks, reduction = True, patch_size = self.reduce_patch_size, mask1 = masks[0][0], mask2 = masks[0][1], roll_matrix = rolls[0]),
            BatchNormTranspose(d_model2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiScaledConvHead(num_heads = res_block2_heads, d_model = d_model2, num_masks = res_block2_masks, reduction = False, mask1 = masks[1][0], mask2 = masks[1][1], roll_matrix = rolls[1]),
            BatchNormTranspose(d_model2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.res_block_22 = nn.Sequential(
            nn.ReLU(),
            MultiScaledConvHead(num_heads = res_block2_heads, d_model = d_model2, num_masks = res_block2_masks, reduction = False, mask1 = masks[1][0], mask2 = masks[1][1], roll_matrix = rolls[1]),
            BatchNormTranspose(d_model2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiScaledConvHead(num_heads = res_block2_heads, d_model = d_model2, num_masks = res_block2_masks, reduction = False, mask1 = masks[1][0], mask2 = masks[1][1], roll_matrix = rolls[1]),
            BatchNormTranspose(d_model2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.res_block_23 = nn.Sequential(
            nn.ReLU(),
            MultiScaledConvHead(num_heads = res_block2_heads, d_model = d_model2, num_masks = res_block2_masks, reduction = False, mask1 = masks[1][0], mask2 = masks[1][1], roll_matrix = rolls[1]),
            BatchNormTranspose(d_model2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiScaledConvHead(num_heads = res_block2_heads, d_model = d_model2, num_masks = res_block2_masks, reduction = False, mask1 = masks[1][0], mask2 = masks[1][1], roll_matrix = rolls[1]),
            BatchNormTranspose(d_model2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.shortcut_projection2 = MultiScaledConvHead(num_heads = res_block2_heads, d_model = d_model2, num_masks = res_block2_masks, reduction = True, patch_size = self.reduce_patch_size, mask1 = masks[1][0], mask2 = masks[1][1], roll_matrix = rolls[1])
        self.res_block_31 = nn.Sequential(
            nn.ReLU(),
            MultiScaledConvHead(num_heads = res_block2_heads, d_model = d_model2, num_masks = res_block2_masks, reduction = True, patch_size = self.reduce_patch_size, mask1 = masks[1][0], mask2 = masks[1][1], roll_matrix = rolls[1]),
            BatchNormTranspose(d_model3),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiScaledConvHead(num_heads = res_block3_heads, d_model = d_model3, num_masks = res_block3_masks, reduction = False, mask1 = masks[2][0], mask2 = masks[2][1], roll_matrix = rolls[2]),
            BatchNormTranspose(d_model3),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.res_block_32 = nn.Sequential(
            nn.ReLU(),
            MultiScaledConvHead(num_heads = res_block3_heads, d_model = d_model3, num_masks = res_block3_masks, reduction = False, mask1 = masks[2][0], mask2 = masks[2][1], roll_matrix = rolls[2]),
            BatchNormTranspose(d_model3),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiScaledConvHead(num_heads = res_block3_heads, d_model = d_model3, num_masks = res_block3_masks, reduction = False, mask1 = masks[2][0], mask2 = masks[2][1], roll_matrix = rolls[2]),
            BatchNormTranspose(d_model3),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.res_block_33 = nn.Sequential(
            nn.ReLU(),
            MultiScaledConvHead(num_heads = res_block3_heads, d_model = d_model3, num_masks = res_block3_masks, reduction = False, mask1 = masks[2][0], mask2 = masks[2][1], roll_matrix = rolls[2]),
            BatchNormTranspose(d_model3),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiScaledConvHead(num_heads = res_block3_heads, d_model = d_model3, num_masks = res_block3_masks, reduction = False, mask1 = masks[2][0], mask2 = masks[2][1], roll_matrix = rolls[2]),
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
# model = ConvFormerResNet(img_h = 32, img_w = 32, patch_size=2).to(mps_device)
# conv_input = torch.stack([CIFAR_Conv_Val.__getitem__(random.randint(0, 9999))[0] for i in range(1)]).to(mps_device)
# print(model(conv_input))

"""
Implementation of Reduced-Size ConvFormers allowing for comparisons with ResNets which have similar amounts of parameters.
"""
class ConvFormerResNetSmall(nn.Module):
    def __init__(self, img_h, img_w, patch_size, d_model = 32, res_heads = 16, mask_fidelity = 3, dropout_p = 0.01):
        super().__init__()
        self.patch_size = patch_size
        self.reduce_patch_size = 2
        img_sizes = [(img_h, img_w), (img_h/(self.reduce_patch_size), img_w/(self.reduce_patch_size)), (img_h/(self.reduce_patch_size**2), img_w/(self.reduce_patch_size**2))] 
        rolls, masks = gen_rolls_and_masks(img_sizes=img_sizes, patch_size = patch_size, filt_type = "filter", mask_fidelity = mask_fidelity, should_train = True)
        multiplier = 2
        res_block1_heads = res_heads
        res_block2_heads = res_block1_heads*multiplier
        res_block3_heads = res_block2_heads*multiplier
        res_block1_masks = res_heads
        res_block2_masks = res_block1_masks*multiplier
        res_block3_masks = res_block2_masks*multiplier
        #Should be square number
        patch_size = 2
        d_model1 = d_model
        d_model2 = d_model1*(self.reduce_patch_size**2)
        d_model3 = d_model2*(self.reduce_patch_size**2)
        dropout_p = dropout_p
        self.conv_scale_up = nn.Sequential(
            nn.Conv2d(3,4,3,padding=1),
            Patchify(patch_size = self.patch_size),
            PositionalEncodingAppend(d_model1, device = get_device()),
        )
        self.res_block_11 = nn.Sequential(
            MultiScaledConvHead(num_heads = res_block1_heads, d_model = d_model1, num_masks = res_block1_masks, reduction = False, mask1 = masks[0][0], mask2 = masks[0][1], roll_matrix = rolls[0]),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.res_block_12 = nn.Sequential(
            nn.ReLU(),
            MultiScaledConvHead(num_heads = res_block1_heads, d_model = d_model1, num_masks = res_block1_masks, reduction = False, mask1 = masks[0][0], mask2 = masks[0][1], roll_matrix = rolls[0]),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiScaledConvHead(num_heads = res_block1_heads, d_model = d_model1, num_masks = res_block1_masks, reduction = False, mask1 = masks[0][0], mask2 = masks[0][1], roll_matrix = rolls[0]),
            BatchNormTranspose(d_model1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.shortcut_projection1 = MultiScaledConvHead(num_heads = res_block1_heads, d_model = d_model1, num_masks = res_block1_masks, reduction = True, patch_size = self.reduce_patch_size, mask1 = masks[0][0], mask2 = masks[0][1], roll_matrix = rolls[0])
        self.res_block_21 = nn.Sequential(
            nn.ReLU(),
            MultiScaledConvHead(num_heads = res_block1_heads, d_model = d_model1, num_masks = res_block1_masks, reduction = True, patch_size = self.reduce_patch_size, mask1 = masks[0][0], mask2 = masks[0][1], roll_matrix = rolls[0]),
            BatchNormTranspose(d_model2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiScaledConvHead(num_heads = res_block2_heads, d_model = d_model2, num_masks = res_block2_masks, reduction = False, mask1 = masks[1][0], mask2 = masks[1][1], roll_matrix = rolls[1]),
            BatchNormTranspose(d_model2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.shortcut_projection2 = MultiScaledConvHead(num_heads = res_block2_heads, d_model = d_model2, num_masks = res_block2_masks, reduction = True, patch_size = self.reduce_patch_size, mask1 = masks[1][0], mask2 = masks[1][1], roll_matrix = rolls[1])
        self.res_block_31 = nn.Sequential(
            nn.ReLU(),
            MultiScaledConvHead(num_heads = res_block2_heads, d_model = d_model2, num_masks = res_block2_masks, reduction = True, patch_size = self.reduce_patch_size, mask1 = masks[1][0], mask2 = masks[1][1], roll_matrix = rolls[1]),
            BatchNormTranspose(d_model3),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            MultiScaledConvHead(num_heads = res_block3_heads, d_model = d_model3, num_masks = res_block3_masks, reduction = False, mask1 = masks[2][0], mask2 = masks[2][1], roll_matrix = rolls[2]),
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
# model = ConvFormerResNetSmall(img_h = 32, img_w = 32, patch_size=1).to(mps_device)
# conv_input = torch.stack([CIFAR_Conv_Val.__getitem__(random.randint(0, 9999))[0] for i in range(1)]).to(mps_device)
# print(model(conv_input))

