import torch.nn as nn
import torchvision.models as models

# Base results / Accuracy:0.64
class Vgg16_FCN32s(nn.Module):
    def __init__(self):
        super(Vgg16_FCN32s, self).__init__()
        self.vgg16_feature = models.vgg16(pretrained=True).features
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 7,  kernel_size=(1, 1), stride=(1, 1))
        )
        self.conv2 = nn.ConvTranspose2d(7, 7, kernel_size=(64, 64), stride=(32, 32), bias=False)
    
    def forward(self, x):
        x = self.vgg16_feature(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# Improved results / Accuracy:0.69
class Vgg19_FCN32s(nn.Module):
    def __init__(self):
        super(Vgg19_FCN32s, self).__init__()
        self.vgg19_feature = models.vgg19(pretrained=True).features
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 7, kernel_size=(1, 1), stride=(1, 1))
        )
        self.conv2 = nn.ConvTranspose2d(7, 7, kernel_size=(64, 64), stride=(32, 32), bias=False)

    def forward(self, x):
        x = self.vgg19_feature(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# Worse results / Accuracy:0.39
class Vgg16_FCN8s(nn.Module):
    def __init__(self):
        super(Vgg16_FCN8s, self).__init__()
        self.vgg16_feature = models.vgg16(pretrained=True).features
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2))
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2))
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, 7, kernel_size=(1, 1))

    def forward(self, x):
        x_8 = self.vgg16_feature[:17](x)                    # (7, 512, 64, 64)
        x_16 = self.vgg16_feature[17:24](x_8)               # (7, 512, 32, 32)
        x_32 = self.vgg16_feature[24:](x_16)                # (7, 512, 16, 16)

        x = self.relu(self.deconv1(x_32))               # (7, 512, 32, 32)
        x = self.bn1(x + x_16)                      # (7, 512, 32, 32)
        x = self.relu(self.deconv2(x))              # (7, 256, 64, 64)
        x = self.bn2(x + x_8)                       # (7, 256, 64, 64)
        x = self.bn3(self.relu(self.deconv3(x)))    # (7, 128, 128, 128)
        x = self.bn4(self.relu(self.deconv4(x)))    # (7, 64, 256, 256)
        x = self.bn5(self.relu(self.deconv5(x)))    # (7, 32, 512, 512)
        x = self.classifier(x)                      # (7, 7, 512, 512)

        return x



