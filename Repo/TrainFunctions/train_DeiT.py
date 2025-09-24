# Import necessary libraries
import os
from torcheval.metrics import MulticlassAccuracy
import os 
from torcheval.metrics import MulticlassAccuracy
import torch
from VisionTransformer.ViT import ViT
from utils.device_utils import get_device
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.make_datasets import return_datasets_IN
import ray
import random
from torchvision.transforms import v2
from functools import partial
import timm
from timm.scheduler import create_scheduler_v2
import time
"""Train DeiT using timm configuration. Not used in paper, since not possible to ablate on LMSA layer"""
def train_DeiT(config, img_h, img_w, patch_size, d_model, tuning_mode = False, sim_batch_size=None):
    batch_size = 1024
    train, val, test = return_datasets_IN(batch_size=batch_size)
    size = len(train)
    print(size)
    epochs = 300
    if sim_batch_size == None:
        sim_batch_size = batch_size
    sim_batch_size = 1024
    mult = sim_batch_size//batch_size
    device = get_device()
    model = timm.create_model('deit_tiny_patch16_224', pretrained = False, num_classes=1000).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters in ViT: {total_params}")
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=config['weight_decay'], lr = config['lr'], eps = 1e-8)
    if config['lr_scheduler'] == 'Warmup-Cosine-Annealing':
        lr_scheduler, _ = create_scheduler_v2(
            sched = 'cosine',
            num_epochs = epochs,
            min_lr = 1e-5,
            warmup_lr = 1e-6,
            warmup_epochs = 5,
            cooldown_epochs = 10,
            optimizer = optimizer
        )
    if(config['lr_scheduler'] == 'OneCycleLR'):
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], epochs = epochs, steps_per_epoch = int(size/sim_batch_size), cycle_momentum=False)
    loss_fn = partial(torch.nn.functional.cross_entropy, label_smoothing=0.1)
    accuracy_fn = MulticlassAccuracy()
    # clip_val = 3
    val_accuracy = []
    model_loss = []
    val_loss = []
    grad_mags = []
    cut_mix = v2.CutMix(num_classes = 1000, alpha = 1.0)
    mixup = v2.MixUp(num_classes = 1000, alpha = 0.8)
    cut_mix_or_mixup = v2.RandomChoice([cut_mix, mixup])
    if not os.path.exists("/workspace/ConvFormer/Repo/models/VisionTransformer"):
        os.mkdir("/workspace/ConvFormer/Repo/models/VisionTransformer")
    for j in range(epochs):
        model_sub_loss = torch.zeros((len(train),))
        model_sub_loss.requires_grad = False
        start = time.perf_counter()
        for i, batch in enumerate(train):
            inputs = batch[0]
            labels = batch[1]
            inputs, labels = cut_mix_or_mixup(inputs, labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs).contiguous()
            loss = loss_fn(outputs, labels)/mult
            model_sub_loss[i] = loss.cpu().detach()/mult
            if(i%(100*mult)==0):
                accuracy_fn.update(outputs, torch.argmax(labels, dim = -1))
                # accuracy_fn.update(outputs, labels)
                print(f"Loss: {loss}, Batch Num: {i//mult}/{len(train)//mult}, Accuracy:{accuracy_fn.compute()}, Epoch: {j}")
            loss.backward()
            if(i%(100*mult) == 0):
                grad_mag = torch.norm(torch.stack([torch.norm(p.grad, 2.0) for p in model.parameters() if p.grad is not None]), 2.0)
                grad_mags.append(grad_mag)
                print(grad_mag)
                torch.cuda.empty_cache()
            if ((i+1)%mult == 0):
                # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                optimizer.step()
                optimizer.zero_grad()
        model_loss.append(torch.mean(model_sub_loss,dim=-1))
        print(f'Time for Epoch: {time.perf_counter()-start}')
        lr_scheduler.step(j)
        print("Validation Stage:")
        dropout_modules = [module for module in model.modules() if isinstance(module,torch.nn.Dropout)]
        model.eval()
        accuracy_fn.reset()
        k = 0
        val_sub_loss = torch.zeros((len(val),))
        val_sub_loss.requires_grad = False
        for i, batch in enumerate(val):
            with torch.no_grad():
                inputs = batch[0]
                labels = batch[1]
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_sub_loss[k] = loss.cpu().detach()
                accuracy_fn.update(outputs, labels)
                if(k%10==0):
                    print(labels)
                    print(f"Loss: {loss}, Batch Num: {i}/{len(val)}, Accuracy:{accuracy_fn.compute()}, Epoch: {j}")
                k+=1
        print(f"Final Accuracy: {accuracy_fn.compute()}")
        
        val_loss.append(torch.mean(val_sub_loss))
        val_accuracy.append(accuracy_fn.compute())
        if tuning_mode:
            ray.train.report({"loss":torch.mean(val_sub_loss).numpy().item(), "accuracy":accuracy_fn.compute().numpy().item()})
        accuracy_fn.reset()
        if not tuning_mode:
            plt.plot(model_loss)
            plt.plot(val_loss)
            plt.show()
            plt.plot(val_accuracy)
            plt.show()
            plt.imshow(inputs[0].permute(1,2,0).cpu().detach())
            plt.show()
        model.train()
        torch.save(model.state_dict(),f"/workspace/ConvFormer/Repo/models/VisionTransformer/model{j}.pt")
    """Save ViT metrics"""
    with open("/workspace/ConvFormer/Repo/models/VisionTransformer/grad_mags.txt", 'w+') as writer:
        for grad_mag in grad_mags:
            writer.write(f"{grad_mag},")
    with open("/workspace/ConvFormer/Repo/models/VisionTransformer/val_loss.txt", 'w+') as writer:
        for loss in val_loss:
            writer.write(f"{loss},")
    with open("/workspace/ConvFormer/Repo/models/VisionTransformer/train_loss.txt", 'w+') as writer:
        for loss in model_loss:
            writer.write(f"{loss},")
    with open("/workspace/ConvFormer/Repo/models/VisionTransformer/val_accuracy.txt", 'w+') as writer:
        for accuracy in val_accuracy:
            writer.write(f"{accuracy},")