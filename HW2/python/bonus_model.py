import torch
import torch.nn as nn
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()

        # input image size: [3, 28, 28]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 0),        # (64,24,24)
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                  # (64,12,12)
            nn.ReLU(True),
            nn.Conv2d(64, 50, 5, 1, 0),       # (64,8,8)
            nn.BatchNorm2d(50),
            nn.MaxPool2d(2),                  # (50,4,4)
            nn.ReLU(True)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(50 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.ReLU(True)
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(512, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def encode(self, x):
        feature = self.cnn_layers(x)
        feature = feature.view(-1, 50 * 4 * 4)
        feature = self.fc_layer(feature)
        return feature

    def forward(self, x, alpha=1.0):
        feature = self.cnn_layers(x)
        feature = feature.view(-1, 50 * 4 * 4)
        feature = self.fc_layer(feature)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

