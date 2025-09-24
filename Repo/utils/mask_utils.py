import torch
import numpy as np

"""
Generate a low-rank attention map for a scpecified filter. Not used in paper.
"""
def gen_filter_attention(img_h, img_w, filter, stride = 1, padding = 1, add = 0.0, append_cls = False):
    filter_attention_matrix = torch.zeros((img_h*img_w,img_h*img_w))+add
    rolls = torch.linspace(1, img_w*img_h, img_w*img_h, dtype = torch.int64).repeat(img_h*img_w,1)
    offset_x = 1-padding
    offset_y = 1-padding
    calc_height = int((img_h+padding*2 - filter.shape[0]+1)/stride)
    calc_width = int((img_w+padding*2 - filter.shape[1]+1)/stride)
    if(calc_width > 0 and calc_height > 0):
        print(calc_height)
        print(calc_width)
        for i in range(calc_height):
            for j in range(calc_width):
                if offset_y == 0:
                    if offset_x == 0:
                        filter_attention_matrix[j+i*calc_width][0:2] = filter[1][1:3]
                        filter_attention_matrix[j+i*calc_width][img_w:img_w+2] = filter[2][1:3]
                    elif offset_x == img_w-1:
                        filter_attention_matrix[j+i*calc_width][offset_x-1:offset_x+1] = filter[1][0:2]
                        filter_attention_matrix[j+i*calc_width][img_w+offset_x-1:img_w+offset_x+1] = filter[2][0:2]
                    else:
                        filter_attention_matrix[j+i*calc_width][offset_x-1:offset_x+2] = filter[1][0:3]
                        filter_attention_matrix[j+i*calc_width][img_w+offset_x-1:img_w+offset_x+2] = filter[1][0:3]
                elif offset_y == img_w-1:
                    if offset_x == 0:
                        filter_attention_matrix[j+i*calc_width][(offset_y-1)*img_w:(offset_y-1)*img_w+2] = filter[0][1:3]
                        filter_attention_matrix[j+i*calc_width][(offset_y)*img_w:(offset_y)*img_w+2] = filter[1][1:3]
                    elif offset_x == img_w-1:
                        filter_attention_matrix[j+i*calc_width][(offset_y-1)*img_w+offset_x-1:(offset_y-1)*img_w+offset_x+1] = filter[0][0:2]
                        filter_attention_matrix[j+i*calc_width][(offset_y)*img_w+offset_x-1:(offset_y)*img_w+offset_x+1] = filter[1][0:2]
                    else:
                        filter_attention_matrix[j+i*calc_width][(offset_y-1)*img_w+offset_x-1:(offset_y-1)*img_w+offset_x+2] = filter[0][0:3]
                        filter_attention_matrix[j+i*calc_width][(offset_y)*img_w+offset_x-1:(offset_y)*img_w+offset_x+2] = filter[1][0:3]
                else:
                    if offset_x == 0:
                        filter_attention_matrix[j+i*calc_width][(offset_y-1)*img_w:(offset_y-1)*img_w+2] = filter[0][1:3]
                        filter_attention_matrix[j+i*calc_width][offset_y*img_w:offset_y*img_w+2] = filter[1][1:3]
                        filter_attention_matrix[j+i*calc_width][(offset_y+1)*img_w:(offset_y+1)*img_w+2] = filter[2][1:3]
                    elif offset_x == img_w-1:
                        filter_attention_matrix[j+i*calc_width][(offset_y-1)*img_w+offset_x-1:(offset_y-1)*img_w+offset_x+1] = filter[0][0:2]
                        filter_attention_matrix[j+i*calc_width][(offset_y)*img_w+offset_x-1:(offset_y)*img_w+offset_x+1] = filter[1][0:2]
                        filter_attention_matrix[j+i*calc_width][(offset_y+1)*img_w+offset_x-1:(offset_y+1)*img_w+offset_x+1] = filter[2][0:2]
                    else:
                        filter_attention_matrix[j+i*calc_width][(offset_y-1)*img_w+offset_x-1:(offset_y-1)*img_w+offset_x+2] = filter[0][0:3]
                        filter_attention_matrix[j+i*calc_width][(offset_y)*img_w+offset_x-1:(offset_y)*img_w+offset_x+2] = filter[1][0:3]
                        filter_attention_matrix[j+i*calc_width][(offset_y+1)*img_w+offset_x-1:(offset_y+1)*img_w+offset_x+2] = filter[2][0:3]
                filter_attention_matrix[j+i*calc_width] = torch.roll(filter_attention_matrix[j+i*calc_width],-(offset_x+offset_y*img_h))
                rolls[j+i*calc_width] = torch.roll(rolls[j+i*calc_width],(offset_x+offset_y*img_h))
                offset_x+=stride
            offset_x=0
            offset_y+=stride
        if append_cls:
            cat_filter_x = torch.ones((img_h*img_w, 1))
            cat_filter_y = torch.ones((1, img_h*img_w+1))
            filter_attention_matrix = torch.cat((cat_filter_x, filter_attention_matrix), dim = -1)
            filter_attention_matrix = torch.cat((cat_filter_y, filter_attention_matrix), dim = 0)
            cat_rolls_x = torch.zeros((img_h*img_w, 1), dtype = torch.int64)
            cat_rolls_y = torch.linspace(0, img_w*img_h, img_w*img_h+1, dtype = torch.int64).reshape(1, img_h*img_w+1)
            rolls = torch.cat((cat_rolls_x, rolls), dim = -1)
            rolls = torch.cat((cat_rolls_y, rolls), dim = 0)
    return filter_attention_matrix, rolls

"""
Generate a matrix that has stronger activatoins for nearby pixels in the image, and weaker activations for pixels farther away to induce locality bias. Distance is calcuated by L2 Norm between pixels of image.
"""
def gen_radial(img_h, img_w, scale = 0.5, stride = 1, append_cls = False):
    rolls = torch.linspace(0, img_w*img_h-1, img_w*img_h, dtype = torch.int64).repeat(img_h*img_w,1)
    radial_attention_matrix = torch.zeros((img_h*img_w,img_h*img_w))
    calc_w = int(img_w/stride)
    calc_h = int(img_h/stride)
    x_pos = 0
    y_pos = 0
    for curr_i in range(calc_h):
        for curr_j in range(calc_w):
            curr_matrix = torch.zeros((img_h, img_w))
            for i in range(img_h):
                for j in range(img_w):
                    euc_weight = 1/np.exp(scale*(((y_pos-i)**2+(x_pos-j)**2)**(0.5)))
                    curr_matrix[i][j] = euc_weight
            radial_attention_matrix[curr_i*calc_h+curr_j] = torch.roll(curr_matrix.reshape(-1),-(y_pos*img_w+x_pos))
            rolls[curr_i*calc_h+curr_j] = torch.roll(rolls[curr_i*calc_h+curr_j],(y_pos*img_w+x_pos))
            x_pos+=stride
        x_pos = 0
        y_pos+=stride
    if append_cls:
        cat_filter_x = torch.ones((img_h*img_w, 1))
        cat_filter_y = torch.ones((1, img_h*img_w+1))
        radial_attention_matrix = torch.cat((cat_filter_x, radial_attention_matrix), dim = -1)
        radial_attention_matrix = torch.cat((cat_filter_y, radial_attention_matrix), dim = 0)
        cat_rolls_x = torch.zeros((img_h*img_w, 1), dtype = torch.int64)
        cat_rolls_y = torch.linspace(0, img_w*img_h, img_w*img_h+1, dtype = torch.int64).reshape(1, img_h*img_w+1)
        rolls = torch.cat((cat_rolls_x, rolls), dim = -1)
        rolls = torch.cat((cat_rolls_y, rolls), dim = 0)
    return radial_attention_matrix, rolls

"Generate a matrix of all ones. Induces no initial inductive bias on the Learned Mask Self Attention layer. Not used in the paper"
def gen_ones(img_h, img_w, stride = 1, append_cls = False):
    rolls = torch.linspace(0, img_w*img_h-1, img_w*img_h, dtype = torch.int64).repeat(img_h*img_w,1)
    ones_attention_matrix = torch.normal(1/(img_h*img_w),0,(img_h*img_w,img_h*img_w))
    calc_w = int(img_w/stride)
    calc_h = int(img_h/stride)
    x_pos = 0
    y_pos = 0
    for curr_i in range(calc_h):
        for curr_j in range(calc_w):
            rolls[curr_i*calc_h+curr_j] = torch.roll(rolls[curr_i*calc_h+curr_j],(y_pos*img_w+x_pos))
            x_pos+=stride
        x_pos = 0
        y_pos+=stride
    if append_cls:
        cat_filter_x = torch.ones((img_h*img_w, 1))
        cat_filter_y = torch.ones((1, img_h*img_w+1))
        ones_attention_matrix = torch.cat((cat_filter_x, ones_attention_matrix), dim = -1)
        ones_attention_matrix = torch.cat((cat_filter_y, ones_attention_matrix), dim = 0)
        cat_rolls_x = torch.zeros((img_h*img_w, 1), dtype = torch.int64)
        cat_rolls_y = torch.linspace(0, img_w*img_h, img_w*img_h+1, dtype = torch.int64).reshape(1, img_h*img_w+1)
        rolls = torch.cat((cat_rolls_x, rolls), dim = -1)
        rolls = torch.cat((cat_rolls_y, rolls), dim = 0)
    return ones_attention_matrix, rolls


"""
Generate a matrix that has stronger activatoins for nearby pixels in the image, and weaker activations for pixels farther away to induce locality bias. Distance is calcuated by L1 Norm between pixels of image. Not used in the paper.
"""
def gen_l1(img_h, img_w, scale = 0.5, stride = 1):
    rolls = torch.linspace(0, img_w*img_h-1, img_w*img_h, dtype = torch.int64).repeat(img_h*img_w,1)
    radial_attention_matrix = torch.zeros((img_h*img_w,img_h*img_w))
    l1_attention_matrix = torch.zeros((img_h*img_w, img_h*img_w))
    calc_w = int(img_w/stride)
    calc_h = int(img_h/stride)
    x_pos = 0
    y_pos = 0
    for curr_i in range(calc_h):
        for curr_j in range(calc_w):
            curr_matrix = torch.zeros((img_h, img_w))
            for i in range(img_h):
                for j in range(img_w):
                    euc_weight = 1/np.exp(scale*(abs(y_pos-i)+abs(x_pos-j)))
                    curr_matrix[i][j]=euc_weight
            l1_attention_matrix[curr_i*calc_h+curr_j] = torch.roll(curr_matrix.reshape(-1),-(y_pos*img_w+x_pos))
            rolls[curr_i*calc_h+curr_j] = torch.roll(rolls[curr_i*calc_h+curr_j],(y_pos*img_w+x_pos))
            x_pos+=stride
        x_pos = 0
        y_pos+=stride
    return l1_attention_matrix, rolls