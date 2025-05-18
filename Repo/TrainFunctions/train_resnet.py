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
from utils.make_datasets import return_datasets
import ray


def train_resnet(config, n = 18, tuning_mode = False):
    CIFAR_Train, CIFAR_Val, CIFAR_Test = return_datasets(size=10000, batch_size=180)
    """Training loop for the ResNets. Plots the validation accuracy along with the train and validation loss. Uses Learning Rate Plateau, SGD and weight decay"""
    device = get_device()
    if not os.path.exists("/Users/adithyagiri/Desktop/STS/Repo/models/ResNet"):
        os.mkdir("/Users/adithyagiri/Desktop/STS/Repo/models/ResNet")
    conv_model = ResNet(n=n).to(device)
    optimizer = torch.optim.SGD(conv_model.parameters(), weight_decay = config['weight_decay'], lr = config['lr'], momentum = config['momentum'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10, mode='min')
    loss_fn = nn.CrossEntropyLoss().to(device)
    accuracy_fn = MulticlassAccuracy()
    epochs = 100
    val_accuracy = []
    model_loss = []
    val_loss = []
    grad_mags = []
    clip_val = 3
    for j in range(epochs):
        plt.show()
        model_sub_loss = torch.zeros((len(CIFAR_Train),))
        for i, batch in enumerate(CIFAR_Train):
            optimizer.zero_grad()
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = conv_model(inputs)
            loss = loss_fn(outputs, labels)
            model_sub_loss[i] = loss.cpu().detach()
            if(i%100==0):
                accuracy_fn.update(outputs, labels)
                print(f"Loss: {loss}, Batch Num: {i}/{len(CIFAR_Train)}, Accuracy:{accuracy_fn.compute()}, Epoch: {j}")
            loss.backward()
            if(i%100 == 0):
                grad_mag = torch.norm(torch.stack([torch.norm(p.grad, 2.0) for p in conv_model.parameters() if p.grad is not None]), 2.0)
                grad_mags.append(grad_mag)
                print(grad_mag)
            torch.nn.utils.clip_grad_norm_(conv_model.parameters(), clip_val)
            optimizer.step()
        model_loss.append(torch.mean(model_sub_loss,dim=-1))
        print("Validation Stage:")

        accuracy_fn.reset()
        k = 0
        val_sub_loss = torch.zeros((len(CIFAR_Val),))
        for i, batch in enumerate(CIFAR_Val):
            with torch.no_grad():
                inputs = batch[0]
                labels = batch[1]
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = conv_model(inputs)
                loss = loss_fn(outputs, labels)
                val_sub_loss[k] = loss.cpu().detach()
                accuracy_fn.update(outputs, labels)
                if(k%10==0):
                    print(f"Loss: {loss}, Batch Num: {i}/{len(CIFAR_Val)}, Accuracy:{accuracy_fn.compute()}, Epoch: {j}")
                k+=1
        lr_scheduler.step(torch.mean(val_sub_loss))
        val_loss.append(torch.mean(val_sub_loss))
        print(f"Final Accuracy: {accuracy_fn.compute()}")
        if tuning_mode:
            ray.train.report({"loss":torch.mean(val_sub_loss).numpy().item(), "accuracy":accuracy_fn.compute().numpy().item()})
        val_accuracy.append(accuracy_fn.compute())
        accuracy_fn.reset()
        if not tuning_mode:
            plt.plot(model_loss)
            plt.plot(val_loss)
            plt.show()
            plt.plot(val_accuracy)
            plt.show()
            plt.imshow(inputs[0].permute(1,2,0).cpu().detach())
            plt.show()
        torch.save(conv_model.state_dict(),f"/Users/adithyagiri/Desktop/STS/Repo/models/ResNet/model{j}.pt")
    """Save ResNet metrics"""
    with open("/Users/adithyagiri/Desktop/STS/Repo/models/ResNet/grad_mags.txt", 'w+') as writer:
        for grad_mag in grad_mags:
            writer.write(f"{grad_mag},")
    with open("/Users/adithyagiri/Desktop/STS/Repo/models/ResNet/val_loss.txt", 'w+') as writer:
        for loss in val_loss:
            writer.write(f"{loss},")
    with open("/Users/adithyagiri/Desktop/STS/Repo/models/ResNet/train_loss.txt", 'w+') as writer:
        for loss in model_loss:
            writer.write(f"{loss},")
    with open("/Users/adithyagiri/Desktop/STS/Repo/models/ResNet/val_accuracy.txt", 'w+') as writer:
        for accuracy in val_accuracy:
            writer.write(f"{accuracy},")

