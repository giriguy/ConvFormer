# Import neccesary libraries
import sys
sys.path.append('../')
from TrainFunctions.train_DeiT import train_DeiT

# Set config of model
config = {
    "lr": 0.001,
    "dropout_p": 0.00,
    "weight_decay": 0.05,
    "momentum": 0.99,
    'lr_scheduler': 'Warmup-Cosine-Annealing',
    }

# Execute train function
train_DeiT(config=config, patch_size = 16, img_h = 224, img_w = 224, d_model = 192)