import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(10, 100)                      # (128,1,1)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),      # (512,4,4)
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(512, 256, 1, 1, 0, bias=False),      # (256,4,4)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),      # (128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(128, 64, 2, 2, 1, bias=False),       # (64,14,14)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),         # (3,28,28)
            nn.Tanh()
        )

    def forward(self, noise, label):
        num = self.label_emb(label)
        noise_number = torch.mul(num, noise)
        noise_number = noise_number.view(noise_number.shape[0], 100, 1, 1)
        x = self.main(noise_number)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(# in:
            # input image size: [3, 28, 28]
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),          # (64,14,14)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),        # (128,7,7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),       # (256,7,7)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),       # (512,7,7)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        )
        self.domain_classifier = nn.Sequential(
            nn.Conv2d(512, 1, 7, 1, 0, bias=False),         # (1,1,1)
            nn.Flatten(),
            nn.Sigmoid()
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        domain = self.main(input)
        domain_output = self.domain_classifier(domain)
        class_output = domain.view(domain.shape[0], -1)
        class_output = self.class_classifier(class_output)
        return domain_output, class_output



