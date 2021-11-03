import torch.nn as nn
import torchvision.models as models


class Vgg16_bn(nn.Module):
    def __init__(self):
        super(Vgg16_bn, self).__init__()
        self.vgg16 = models.vgg16_bn(pretrained=True)        
        self.vgg16.classifier[6] = nn.Linear(4096, 50)   # tune the last classifier layer from 1000 to 50 dimension

    def forward(self, x):
        x = self.vgg16(x)
        return x
