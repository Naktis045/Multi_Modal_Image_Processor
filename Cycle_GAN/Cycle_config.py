import os
import torch
import albumentations as A
from PIL import Image

# Hyperparamethers
batch_size = 1 # There is gonna be no Batch split, in the Cycle GAN batch_size is set to 1
num_workers = 4
lr= 2e-4
max_epochs = 200
min_epochs = 100
lambda_identity = 0.0 # As not used in the paper
lambda_cycle = 10

# Tensorboard
load_model = True
save_model = True

checkpoint_gen_A = "Cycle_GAN/checkpoints/gen_A.pth.tar"
checkpoint_gen_B = "Cycle_GAN/checkpoints/gen_B.pth.tar"

# Data
base_dir = os.path.join("Cycle_GAN", "Restyle_dataset")

data_dir_train = os.path.join(base_dir, "train")
data_dir_test = os.path.join(base_dir, "test")

# Device
accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
devices = 1 # or [0,1,...] for mult. GPU 
device = "cuda" if torch.cuda.is_available() else "cpu"

'''transforms = A.Compose(
    [
        A.Resize(width=256, hight=256),
        A.HorizontalFlip(p=0.4),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255),
        A.ToTensorV2
    ]
)'''
