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

# Upsampling/ Downsampling block for the generator
class ConvBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs)
            if down
               # nn.InstanceNorm2d(out_channels),
                # nn.ReLU(inplace=True) if use_act else nn.Identity()# nn.Identity() - just pass through the input without any changes
            else
                nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if use_act else nn.Identity()# nn.Identity() - just pass through the input without any changes
                )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(pl.LightningModule):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1, use_act=False)# As parer says
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(pl.LightningModule):
    def __init__(self, img_channels, num_residuals=9, num_features = 64):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

        # Downsampling
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1)
            ]
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )

        # Upsampling
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(num_features * 2, num_features, down=False, kernel_size=3, stride=2, padding=1, output_padding=1)
            ]
        )# __output_padding=1__ not essential, but ensures that output size is input size *2(as in the paper)

        self.final = nn.Sequential(
            nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.final(x))
    
def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((1, img_channels, img_size, img_size))
    model = Generator(img_channels, num_residuals=9)
    print(model(x).shape)


if __name__ == "__main__":
    test()