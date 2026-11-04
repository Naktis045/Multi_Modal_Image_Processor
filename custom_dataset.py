import os
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

from config import input_size, hidden_size, output_size, num_classes, batch_size, num_workers, data_dir, accelerator, devices

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

"""
standard DataModule overrides a few core hook methods depending on your needs:

--- prepare_data(): Downloads or saves data to disk. Runs on only one process/GPU in distributed settings to avoid duplicate downloads.

---setup(stage): Handles operations that need to run on every GPU (e.g., loading datasets into memory, performing train/val/test splits, applying transformations).

---train_dataloader(): Returns the DataLoader for training data.

---val_dataloader(): Returns the DataLoader for validation data.

---test_dataloader(): Returns the DataLoader for testing data.
"""
