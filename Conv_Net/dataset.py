import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import pytorch_lightning as pl

import torchmetrics as metrics
from torchmetrics import Mertric

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from config import input_size, hidden_size, output_size, num_classes, batch_size, num_workers, data_dir, accelerator, devices

class ConvDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=batch_size, num_workers=num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self, stage):
        # single GPU
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage):
        # Mult GPU
        entire_dataset = datasets.MNIST(
            root = self.data_dir,
            train = True,
            transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.RandomHorizontalFlip(p=0.2),
                #transforms.RandomVerticalFlip(p=0.2), # horrible for MNIST:)
            ]),
            download = False,
                                        )
        self.train_dataset, self.val_dataset = random_split(entire_dataset, [50000, 10000])

        self.test_dataset = datasets.MNIST(
            root = self.data_dir,
            train = False,
            transform = transforms.ToTensor(),
            download = False
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers)
    # can do a pin_memory = True for faster data transfer to GPU if using GPU
    # persistant_workers = True can be used to keep the workers alive for faster data loading in subsequent epochs

        #also - def predict_dataloader --- Inference only ****


"""
standard DataModule overrides a few core hook methods depending on your needs:

--- prepare_data(): Downloads or saves data to disk. Runs on only one process/GPU in distributed settings to avoid duplicate downloads.

---setup(stage): Handles operations that need to run on every GPU (e.g., loading datasets into memory, performing train/val/test splits, applying transformations).

---train_dataloader(): Returns the DataLoader for training data.

---val_dataloader(): Returns the DataLoader for validation data.

---test_dataloader(): Returns the DataLoader for testing data.
"""

# Custom Metric  ************
"""
class My_Accuracy(Mertic):
    def __init__(self,):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")# if multiple GPUs are used, the correct counts from each GPU will be summed up to get the total correct count across all GPUs.
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()
"""