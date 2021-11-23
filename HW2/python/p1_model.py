import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()                       # (128,1,1)
        self.Generator = nn.Sequential(
            nn.ConvTranspose2d(128, 512, 4, 1, 0, bias=False),  # (512,4,4)
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # (256,8,8)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # (128,16,16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),   # (64,32,32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),     # (3,64,64)
            nn.Tanh()
        )

    def forward(self, x):
        return self.Generator(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()                   # (3,64,64)
        self.Discriminator = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),             # (64,32,32)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),           # (128,16,16)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),           # (256,8,8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),          # (512,4,4)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),            # (1,1,1)
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.Discriminator(x)

