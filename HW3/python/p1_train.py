import glob
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from p1_dataset import DATA
from p1_model import ViT

batch = 24
epoch = 60

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

manualSeed = 1004
random.seed(manualSeed)
torch.manual_seed(manualSeed)

if __name__ == "__main__":
    train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    trainset = DATA(root='p1_data/train', transform=train_transform)
    validset = DATA(root='p1_data/val', transform=val_transform)
    trainset_loader = DataLoader(dataset=trainset, batch_size=batch, shuffle=True)
    validset_loader = DataLoader(validset, batch_size=batch, shuffle=False, num_workers=0)

    model = ViT().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epoch):
        for batch, (image, label) in enumerate(trainset_loader, 1):
            image, label = image.to(device), label.to(device)

            output = model(image)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration = 100. * batch / len(trainset_loader)
            if iteration % 25 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                                                        ep,
                                                                        batch * len(image),
                                                                        len(trainset_loader.dataset),
                                                                        iteration,
                                                                        loss.item()))

        val_loss = 0
        correct = 0
        with torch.no_grad():
            for x, label in validset_loader:
                x, label = x.to(device), label.to(device)
                out = model(x)
                val_loss += criterion(out, label).item()
                pred = out.max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(validset_loader.dataset)
        val_loss /= len(validset_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(val_loss,
                                                                                     correct,
                                                                                     len(validset_loader.dataset),
                                                                                     accuracy))
        # Save trained model
        if accuracy >= 94:
            state = {'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            torch.save(state, 'checkpoint/p1-%i.pth' % ep)
            print('Model saved!')
