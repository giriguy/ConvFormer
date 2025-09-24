# Import neccesary libraries
import sys
sys.path.append('../')
from Repo.TrainFunctions.train_IBiT import train_IBiT

# Set config of model
config = {
    "lr": 0.001,
    "dropout_p": 0.00,
    "weight_decay": 0.05*(4/2),
    "momentum": 0.99,
    "mask_fidelity": 9,
    'lr_scheduler': 'Warmup-Cosine-Annealing',
    }

# Execute train function
train_IBiT(config=config, patch_size = 16, img_h = 224, img_w = 224, d_model = 192)
