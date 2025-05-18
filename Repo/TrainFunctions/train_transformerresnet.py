import os
from torcheval.metrics import MulticlassAccuracy
import os 
from torcheval.metrics import MulticlassAccuracy
import torch
from TransformerResNet.TransformerResNet import TransformerResNetSmall
from utils.device_utils import get_device
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.make_datasets import return_datasets
import ray

def train_transformerresnet(config, patch_size, img_h, img_w, d_model, tuning_mode = False):
    CIFAR_Train, CIFAR_Val, CIFAR_Test = return_datasets(size=40000, batch_size=180)
    device = get_device()
    model = TransformerResNetSmall(img_h = img_h, img_w = img_w, patch_size=patch_size, dropout_p = config['dropout_p'], d_model = d_model).to(device)
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=config['weight_decay'], lr = config['lr'], momentum = config['momentum'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 10)
    if not os.path.exists("/Users/adithyagiri/Desktop/STS/Repo/models/TransformerResNet"):
        os.mkdir("/Users/adithyagiri/Desktop/STS/Repo/models/TransformerResNet")
    loss_fn = nn.NLLLoss().to(device)
    accuracy_fn = MulticlassAccuracy()
    epochs = 100
    val_accuracy = []
    model_loss = []
    val_loss = []
    grad_mags = []
    clip_val = 3
    start_batch = 0
    for j in range(epochs):
        model_sub_loss = torch.zeros((len(CIFAR_Train),))
        model_sub_loss.requires_grad = False
        for i, batch in enumerate(CIFAR_Train):
            if j == 0 and i < start_batch:
                continue
            optimizer.zero_grad(set_to_none=True)
            inputs = batch[0]
            labels = batch[1]
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            model_sub_loss[i] = loss.cpu().detach()
            if(i%100==0):
                accuracy_fn.update(outputs, labels)
                print(f"Loss: {loss}, Batch Num: {i}/{len(CIFAR_Train)}, Accuracy:{accuracy_fn.compute()}, Epoch: {j}")
            loss.backward()
            if(i%100 == 0):
                grad_mag = torch.norm(torch.stack([torch.norm(p.grad, 2.0) for p in model.parameters() if p.grad is not None]), 2.0)
                grad_mags.append(grad_mag)
                print(grad_mag)
                torch.cuda.empty_cache()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            optimizer.step()
        model_loss.append(torch.mean(model_sub_loss,dim=-1))
        print("Validation Stage:")
        dropout_modules = [module for module in model.modules() if isinstance(module,torch.nn.Dropout)]
        [module.eval() for module in dropout_modules]
        accuracy_fn.reset()
        k = 0
        val_sub_loss = torch.zeros((len(CIFAR_Val),))
        val_sub_loss.requires_grad = False
        for i, batch in enumerate(CIFAR_Val):
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
                    print(f"Loss: {loss}, Batch Num: {i}/{len(CIFAR_Val)}, Accuracy:{accuracy_fn.compute()}, Epoch: {j}")
                k+=1
        print(f"Final Accuracy: {accuracy_fn.compute()}")
        lr_scheduler.step(torch.mean(val_sub_loss))
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
        print(f"lr: {lr_scheduler._last_lr }")
        [module.train() for module in dropout_modules]
        torch.save(model.state_dict(),f"/Users/adithyagiri/Desktop/STS/Repo/models/TransformerResNet/model{j}.pt")
    """Save TransformerResNet metrics"""
    with open("/Users/adithyagiri/Desktop/STS/Repo/models/TransformerResNet/grad_mags.txt", 'w+') as writer:
        for grad_mag in grad_mags:
            writer.write(f"{grad_mag},")
    with open("/Users/adithyagiri/Desktop/STS/Repo/models/TransformerResNet/val_loss.txt", 'w+') as writer:
        for loss in val_loss:
            writer.write(f"{loss},")
    with open("/Users/adithyagiri/Desktop/STS/Repo/models/TransformerResNet/train_loss.txt", 'w+') as writer:
        for loss in model_loss:
            writer.write(f"{loss},")
    with open("/Users/adithyagiri/Desktop/STS/Repo/models/TransformerResNet/val_accuracy.txt", 'w+') as writer:
        for accuracy in val_accuracy:
            writer.write(f"{accuracy},")
