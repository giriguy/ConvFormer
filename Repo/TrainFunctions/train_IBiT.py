# Import necessary libraries
import os 
from torcheval.metrics import MulticlassAccuracy
import torch
from IBiT.IBiT import IBiT
from utils.device_utils import get_device
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.make_datasets import return_datasets_IN
import ray
from torchvision.transforms import v2
from functools import partial
import time
import timm
from timm.scheduler import create_scheduler_v2

# Train function for IBiT
def train_IBiT(config, img_h, img_w, patch_size, d_model, tuning_mode = False, sim_batch_size = None):
    
    # Initialize dataset and model parameters. Use simulated batch size for equal comparisons.
    model_loc = "/workspace/IBiT/Repo"
    save_name = "IBiT"
    batch_size = 512
    train, val, test = return_datasets_IN(batch_size=batch_size)
    size = len(train)
    epochs = 300
    if sim_batch_size == None:
        sim_batch_size = batch_size
    sim_batch_size = 1024
    mult = sim_batch_size//batch_size
    
    # Get device and initialize learning rate scheduler.
    device = get_device()
    model = IBiT(img_h = img_h, img_w = img_w, d_model = d_model, patch_size=patch_size, mask_fidelity=config['mask_fidelity'], dropout_p=config['dropout_p']).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters in CiT: {total_params}")
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
    
    # Create loss function and arrays to log model accuracy
    loss_fn = partial(torch.nn.functional.cross_entropy, label_smoothing=0.1)
    accuracy_fn = MulticlassAccuracy()
    clip_val = 3
    val_accuracy = []
    model_loss = []
    val_loss = []
    grad_mags = []

    # The final model didn't include the use of CutMix. Included here for testing purposes
    # cut_mix = v2.CutMix(num_classes = 1000, alpha = 1.0)
    # mixup = v2.MixUp(num_classes = 1000, alpha = 0.8)
    # cut_mix_or_mixup = v2.RandomChoice([cut_mix, mixup])

    # Create model directory to store logs and models
    if not os.path.exists(f"{model_loc}/models/{save_name}"):
        os.mkdir(f"{model_loc}/models/{save_name}")

    # Training Loop 
    for j in range(epochs):
        # Store losses for averaging later
        model_sub_loss = torch.zeros((len(train),))
        model_sub_loss.requires_grad = False
        start = time.perf_counter()
        for i, batch in enumerate(train):

            # Read input data
            inputs = batch[0]
            labels = batch[1]

            # Cutmix version. Included for testing purposes
            # inputs, labels = cut_mix_or_mixup(inputs, labels)

            # Move model to device and compute loss and gradients 
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)/mult
            model_sub_loss[i] = loss.cpu().detach()/mult
            loss.backward()

            # Log accuracy and other metrics
            if(i%(100*mult) == 0):
                
                # Cutmix version. Included for testing purposes
                # accuracy_fn.update(outputs, torch.argmax(labels, dim = -1))
                
                # Compute Accuracy
                accuracy_fn.update(outputs, labels)
                
                # Log model training data
                print(f"Loss: {loss}, Batch Num: {i//mult}/{len(train)//mult}, Accuracy:{accuracy_fn.compute()}, Epoch: {j}, Time: {time.perf_counter()-start}")
                grad_mag = torch.norm(torch.stack([torch.norm(p.grad, 2.0) for p in model.parameters() if p.grad is not None]), 2.0)
                grad_mags.append(grad_mag)
                print(grad_mag)
                torch.cuda.empty_cache()

            if ((i+1)%mult == 0):
                # Gradient clipping is not used in the final model. Included for testing purposes
                # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        
                # Update model
                optimizer.step()
                optimizer.zero_grad()

        # Log time and other metrics and prepare model for evaluation on validation set
        print(f'Time for Epoch: {time.perf_counter()-start}')
        model_loss.append(torch.mean(model_sub_loss,dim=-1))
        lr_scheduler.step(j)
        print("Validation Stage:")
        model.eval()
        print(len(val), batch_size)
        accuracy_fn.reset()
        k = 0
        val_sub_loss = torch.zeros((len(val),))
        val_sub_loss.requires_grad = False

        # Validation loop for model
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
        
        # Add validation scores and other metrics to recording arrays
        val_loss.append(torch.mean(val_sub_loss))
        val_accuracy.append(accuracy_fn.compute())

        # Plot figures if not hyperparameter tuning model
        if tuning_mode:
            ray.train.report({"loss":torch.mean(val_sub_loss).numpy().item(), "accuracy":accuracy_fn.compute().numpy().item()})
        accuracy_fn.reset()
        if not tuning_mode:
            plt.plot(val_accuracy)
            plt.show()

        # Put model back in training mode and save model 
        model.train()
        torch.save(model.state_dict(),f"/{model_loc}/models/{save_name}/model{j}.pt")
    
    
    # Save model and metrics of final model
    with open(f"{model_loc}/models/{save_name}/grad_mags.txt", 'w+') as writer:
        for grad_mag in grad_mags:
            writer.write(f"{grad_mag},")
    with open(f"{model_loc}/models/{save_name}/val_loss.txt", 'w+') as writer:
        for loss in val_loss:
            writer.write(f"{loss},")
    with open(f"{model_loc}/models/{save_name}/train_loss.txt", 'w+') as writer:
        for loss in model_loss:
            writer.write(f"{loss},")
    with open(f"{model_loc}/models/{save_name}/val_accuracy.txt", 'w+') as writer:
        for accuracy in val_accuracy:
            writer.write(f"{accuracy},")