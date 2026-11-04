import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

import torchmetrics as metrics
from torchmetrics import Metric
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from config import input_size, hidden_size, output_size, num_classes, batch_size, num_workers, data_dir_train, data_dir_test, accelerator, devices, learning_rate


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, discriminator=False, use_act=True, use_bn=True):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))


class Upsample(nn.Module):
    def __init__(self, in_c, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c * scale_factor ** 2, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class Residual(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = Conv(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.block2 = Conv(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False,
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=16):
        super().__init__()
        self.initial = Conv(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residual = nn.Sequential(*[Residual(num_channels) for _ in range(num_blocks)])
        self.convblock = Conv(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.upsample = nn.Sequential(Upsample(num_channels, scale_factor=2), Upsample(num_channels, scale_factor=2))
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residual(initial)
        x = self.convblock(x)
        x = self.upsample(x)
        return torch.tanh(self.final(x))
    

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        block = []
        for idx, feature in enumerate(features):
            block.append(
                Conv(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx == 0 else True,
                )
            )
            in_channels = feature

        self.block = nn.Sequential(*block)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.block(x)
        return self.classifier(x)

    def early_stopping():
        pass


def test():
    low_res = 24
    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
        x = torch.randn((5, 3, low_res, low_res))
        gen = Generator()
        gen_out = gen(x)
        disc = Discriminator()
        disc_out = disc(gen_out)

        print("Generator output shape:", gen_out.shape)
        print("Discriminator output shape:", disc_out.shape)

if __name__ == '__main__':
    test()