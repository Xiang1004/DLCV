import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SSL(nn.Module):
    def __init__(self):
        super().__init__()
        # self.resnet50 = models.resnet50(pretrained=False)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, 65),
        )

    def forward(self, x):
        # x = self.resnet50(x)
        x = self.classifier(x)
        return x


