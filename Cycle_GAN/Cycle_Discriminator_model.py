import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import torchmetrics
from torchmetrics import Metric

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

# Tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f'runs/CycleGAN/tensorboard')
step = 0

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(padding=1),#) to cleanly halve the spatial dimensions of image feature_maps at every layer.
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding_mode='reflect', bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),# inplace = True means that the input will be modified directly,
                # without allocating new memory for the output. This can save memory and improve performance,
                # but it also means that the original input will be lost after the operation.
                # If you need to keep the original input for later use, you should set inplace=False.
            nn.ReflectionPad2d(padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=stride, padding_mode='reflect', bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(pl.LightningModule):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature==features[-1] else 2))
            # Basically we are gonna use stride=1 for the last layer, and stride=2 for all the other layers
            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')# (1 -> in_channels (NOT RGB))
        )# Output value between (0,1) - indicates whether the image is real or fake
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x)) #   sigmoid -> for prob. between (0,1)

def test():
    x = torch.randn(5, 3, 256, 256)
    model = Discriminator()
    output = model(x)
    print(output.shape)

if __name__ == "__main__":
    test()
