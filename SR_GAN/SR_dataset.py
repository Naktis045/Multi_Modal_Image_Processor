import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

import torchmetrics as metrics
from torchmetrics import Metric
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
from config import input_size, hidden_size, output_size, num_classes, batch_size, num_workers, data_dir_train, data_dir_test, accelerator, devices, learning_rate

import albumentations as A
import cv2 as cv2
from albumentations.pytorch import ToTensorV2


class SRDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "*.*")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image # if there is any label create __y_label__ and return it as well


class SRDataModule(pl.LightningDataModule):
    def __init__(self, data_dir_train=data_dir_train, data_dir_test=data_dir_test, batch_size=batch_size, num_workers=num_workers, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.train_data_dir = data_dir_train
        self.test_data_dir = data_dir_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs #container for any extra configuration options that aren't explicitly defined
# (such as scale_factor=4, patch_size=96, or image_size=256)
# Saving Hyperparameters: If you add self.save_hyperparameters() inside your __init__, Lightning will automatically log all your initialization arguments (including everything packed into **kwargs) to TensorBoard, Weights & Biases
    
    def prepare_data(self):
        if not os.path.exists(self.train_data_dir):
            print(f" Training directory '{self.train_data_dir}' not found")
        
        if not os.path.exists(self.test_data_dir):
            print(f" Test directory '{self.test_data_dir}' not found")

    def setup(self, stage=None):
        train_transform = A.Compose([
            A.Resize(height=720, width=1280),
            A.VerticalFlip(p=0.2),
            A.HorizontalFlip(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.RandomBrightnessContrast(p=0.2),
            A.OneOf([
                A.ColorJitter(saturation=0.2, brightness=0.2, contrast=0.2, hue=0.2, p=0.2),
                A.Blur(p=0.2)
            ]),
            ToTensorV2(),
        ])

        val_transform = A.Compose([
            A.Resize(height=720, width=1280),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.Blur(p=0.2),
            ToTensorV2(),
        ])

        test_transform = A.Compose([
            A.Resize(height=720, width=1280),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.Blur(p=0.2),
            ToTensorV2(),
        ])

        if stage == 'fit' or stage is None:
            full_train_dataset = SRDataset(self.train_data_dir, transform=train_transform)
            train_size = int(0.8 * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size


            self.train_dataset, self.val_dataset = random_split(full_train_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            self.test_dataset = SRDataset(self.test_data_dir, transform=test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )