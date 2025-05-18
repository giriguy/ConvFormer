import torch
import torch.nn as nn
from utils.mask_trainers import MaskTrainer, DoubleMaskTrainer
from utils.mask_utils import gen_filter_attention, gen_radial, gen_ones, gen_l1
from utils.device_utils import get_device
import matplotlib.pyplot as plt
import sys 
"""
Function that executes training procedure for the Learnable Masks using MaskTrainer, generating the trained masks that mimic the filter attention map, and are used for rank approximation.
"""
def gen_rolls_and_masks(img_sizes, patch_size, filt_type = 'filter', should_train = True, add_cls_embeddding = False, mask_fidelity = 9):
    device = get_device()
    img_sizes = [(int(img_h/patch_size), int(img_w/patch_size)) for img_h, img_w in img_sizes]
    rolls = []
    masks = []
    for j, (img_h, img_w) in enumerate(img_sizes):
        filter = torch.normal(1, 0, size = (3,3))
        if filt_type == 'filter':
            print(f"img_h = {img_h} + img_w = {img_w}")
            if add_cls_embeddding:
                attention_matrix, roll_data = (data.to(device) for data in gen_filter_attention(filter=filter, img_h = img_h, img_w = img_w, stride = 1, append_cls=True))
            else:
                attention_matrix, roll_data = (data.to(device) for data in gen_filter_attention(filter=filter, img_h = img_h, img_w = img_w, stride = 1))
            rolls.append(roll_data)
        elif filt_type == 'radial':
            print(f"img_h = {img_h} + img_w = {img_w}")
            attention_matrix, roll_data = (data.to(device) for data in gen_radial(img_h = img_h, img_w = img_w, stride = 1))
            rolls.append(roll_data)
        elif filt_type == 'ones':
            print(f"img_h = {img_h} + img_w = {img_w}")
            attention_matrix, roll_data = (data.to(device) for data in gen_ones(img_h = img_h, img_w = img_w, stride = 1))
            rolls.append(roll_data)
        else:
            print(f"img_h = {img_h} + img_w = {img_w}")
            attention_matrix, roll_data = (data.to(device) for data in gen_l1(img_h = img_h, img_w = img_w, stride = 1))
            rolls.append(roll_data)
        plt.matshow(roll_data.cpu())
        plt.colorbar()
        plt.show()
        model = MaskTrainer(seq_len = img_h*img_w+1, mask_fidelity = mask_fidelity).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
        loss_fn = nn.MSELoss()
        loop_num = 10000
        if(should_train):
            for i in range(loop_num):
                optimizer.zero_grad()
                outputs = model()
                loss = loss_fn(outputs, attention_matrix)
                if i % 2000 == 0:
                    print(loss)
                loss.backward()
                optimizer.step()
        mask1 =  model.mask1
        mask2 = model.mask2
        masks.append((mask1, mask2))
    return (rolls, masks)