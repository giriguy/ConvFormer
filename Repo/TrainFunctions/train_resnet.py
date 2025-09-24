import os
from torcheval.metrics import MulticlassAccuracy
import os 
from torcheval.metrics import MulticlassAccuracy
import torch
from ResNet.ResNet import ResNet
from utils.device_utils import get_device
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.make_datasets import return_datasets_IN
import ray
import torchvision

""" Not used in paper. Implements training loop for ResNets"""
def train_resnet(config, n = 18, tuning_mode = False, sim_batch_size = None):
    size = 121689
    batch_size = 64
    if sim_batch_size == None:
        sim_batch_size = batch_size
    sim_batch_size = 256
    mult = sim_batch_size//batch_size
    epochs = 90
    IN_Train, IN_Val, IN_Test = return_datasets_IN(size=size, batch_size=batch_size)
    device = get_device()
    model = torchvision.models.resnet18()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters in ResNet: {total_params}")
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=config['weight_decay'], lr = config['lr'], momentum = config['momentum'])
    if config['lr_scheduler'] == 'Plateau_W_Warmup':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 10)
    if(config['lr_scheduler'] == 'OneCycleLR'):
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], epochs = epochs, steps_per_epoch = int(size/batch_size), cycle_momentum=False)
    if not os.path.exists("/workspace/ConvFormer/Repo/models/ResNet"):
        os.mkdir("/workspace/ConvFormer/Repo/models/ResNet")
    loss_fn = nn.CrossEntropyLoss().to(device)
    accuracy_fn = MulticlassAccuracy()
    val_accuracy = []
    model_loss = []
    val_loss = []
    grad_mags = []
    clip_val = 3
    for j in range(epochs):
        print("Training Stage:")
        model_sub_loss = torch.zeros((len(IN_Train),))
        for i, batch in enumerate(IN_Train):
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            model_sub_loss[i] = loss.cpu().detach()/mult
            if(i%(100*mult)==0):
                accuracy_fn.update(outputs, labels)
                print(f"Loss: {loss}, Sim Batch Num: {i//mult}/{len(IN_Train)//mult}, Accuracy:{accuracy_fn.compute()}, Epoch: {j}")
            loss.backward()
            if(i%(100*mult) == 0):
                grad_mag = torch.norm(torch.stack([torch.norm(p.grad, 2.0) for p in model.parameters() if p.grad is not None]), 2.0)
                grad_mags.append(grad_mag)
                print(grad_mag)
            if ((i+1)%mult == 0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                optimizer.step()
                optimizer.zero_grad()
        model_loss.append(torch.mean(model_sub_loss,dim=-1))
        print("Validation Stage:")
        accuracy_fn.reset()
        k = 0
        val_sub_loss = torch.zeros((len(IN_Val),))
        for i, batch in enumerate(IN_Val):
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
                    print(f"Loss: {loss}, Batch Num: {i}/{len(IN_Val)}, Accuracy:{accuracy_fn.compute()}, Epoch: {j}")
                k+=1
        lr_scheduler.step(torch.mean(val_sub_loss))
        val_loss.append(torch.mean(val_sub_loss))
        val_accuracy.append(accuracy_fn.compute())
        print(f"Final Accuracy: {accuracy_fn.compute()}")
        if not tuning_mode:
            plt.plot(model_loss)
            plt.plot(val_loss)
            plt.show()
            plt.plot(val_accuracy)
            plt.show()
            plt.imshow(inputs[0].permute(1,2,0).cpu().detach())
            plt.show()
        if tuning_mode:
            ray.train.report({"loss":torch.mean(val_sub_loss).numpy().item(), "accuracy":accuracy_fn.compute().numpy().item()})
        accuracy_fn.reset()
        torch.save(model.state_dict(),f"/workspace/ConvFormer/Repo/models/ResNet/model{j}.pt")
    print("Training Complete")
    """Save ResNet metrics"""
    with open("/workspace/ConvFormer/Repo/models/ResNet/grad_mags.txt", 'w+') as writer:
        for grad_mag in grad_mags:
            writer.write(f"{grad_mag},")
    with open("/workspace/ConvFormer/Repo/models/ResNet/val_loss.txt", 'w+') as writer:
        for loss in val_loss:
            writer.write(f"{loss},")
    with open("/workspace/ConvFormer/Repo/models/ResNet/train_loss.txt", 'w+') as writer:
        for loss in model_loss:
            writer.write(f"{loss},")
    with open("/workspace/ConvFormer/Repo/models/ResNet/val_accuracy.txt", 'w+') as writer:
        for accuracy in val_accuracy:
            writer.write(f"{accuracy},")

