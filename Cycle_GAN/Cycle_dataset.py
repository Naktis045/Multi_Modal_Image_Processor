import os
import numpy as np
import random
from PIL import Image
import torch 
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

import albumentations as A
from albumentations.pytorch import ToTensorV2

from Cycle_config import data_dir_train, data_dir_test, accelerator, devices, device, batch_size, num_workers


class Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.transform = transform
        self.dir_A = os.path.join(root_dir, split, "A")
        self.dir_B = os.path.join(root_dir, split, "B")
        
        self.images_A = sorted(os.listdir(self.dir_A))
        self.images_B = sorted(os.listdir(self.dir_B))
        
    def __len__(self):
        return max(len(self.images_A), len(self.images_B))
        
    def __getitem__(self, index):
        # Get image paths directly
        path_A = os.path.join(self.dir_A, self.images_A[index % len(self.images_A)])
        path_B = os.path.join(self.dir_B, random.choice(self.images_B))
        
        # Load images
        image_A = Image.open(path_A).convert("RGB")
        image_B = Image.open(path_B).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)
            
        return {"A": image_A, "B": image_B}


class CycleDataModule(pl.LightningDataModule):
    def __init__(self, data_dir_train=data_dir_train, data_dir_test=data_dir_test, batch_size=batch_size, num_workers=num_workers, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        
        self.train_data_dir = data_dir_train
        self.test_data_dir = data_dir_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs#container for any extra configuration options that aren't explicitly defined
# (such as scale_factor=4, patch_size=96, or image_size=256)
# Saving Hyperparameters: If you add self.save_hyperparameters() inside your __init__, 
# Lightning will automatically log all your initialization arguments (including everything packed into **kwargs) to TensorBoard, 
# Weights & Biases.
        
    def prepare_data(self):
        if not os.path.exists(self.train_data_dir):
            print(f" Training directory '{self.train_data_dir}' not found")
               
        if not os.path.exists(self.test_data_dir):
            print(f" Test directory '{self.test_data_dir}' not found")

    def setup(self, stage=None):
        train_transform = A.Compose([
            A.Resize(width=256, height=256, interpolation=Image.BICUBIC),
            A.RandomCrop(256, 256),
            A.VerticalFlip(p=0.2),
            A.HorizontalFlip(p=0.2),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            A.RandomBrightnessContrast(p=0.2),
            A.OneOf([
                A.ColorJitter(saturation=0.2, brightness=0.2, contrast=0.2, hue=0.2, p=0.2),
                A.Blur(p=0.2)
            ]),
            ToTensorV2(),
        ])

        val_transform = A.Compose([
            A.Resize(width=256, height=256, interpolation=Image.BICUBIC),
            A.RandomCrop(256, 256),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            A.RandomBrightnessContrast(p=0.2),
            A.Blur(p=0.2),
            ToTensorV2(),
        ])

        test_transform = A.Compose([
            A.Resize(width=256, height=256, interpolation=Image.BICUBIC),
            A.RandomCrop(256, 256),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            A.Blur(p=0.2),
            ToTensorV2(),
        ])

        if stage == 'fit' or stage is None:
            full_train_dataset = Dataset(self.train_data_dir, split="train", transform=train_transform)
            train_size = int(0.8 * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(full_train_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            self.test_dataset = Dataset(self.test_data_dir, split="test", transform=test_transform)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory = True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory = True)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory = True)
