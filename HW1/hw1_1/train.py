import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Vgg16_bn
from dataset import DATA

Input_Size = 224
Batch = 32
# 數值來自 pytorch官網
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# 調整、增強照片
train_transform = transforms.Compose([
    transforms.Resize(Input_Size),  # 調整大小
    transforms.ColorJitter(),  # 調整亮度、對比、飽和...
    transforms.RandomRotation(degrees=10),  # 旋轉角度
    transforms.RandomHorizontalFlip(),  # 水平翻轉
    transforms.ToTensor(),  # 拉直(維度轉換)
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)  # 正規化
])

val_transform = transforms.Compose([
    transforms.Resize(Input_Size),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

# 載入照片
trainset = DATA(root='p1_data/train_50', transform=train_transform)
validset = DATA(root='p1_data/val_50', transform=val_transform)
trainset_loader = DataLoader(trainset, batch_size=Batch, shuffle=True, num_workers=0)
validset_loader = DataLoader(validset, batch_size=Batch, shuffle=False, num_workers=0)

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(100)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)


def training(model, epoch):
    global ep
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()  # set training model

    iteration = 0
    for ep in range(epoch):
        for batch, (x, label) in enumerate(trainset_loader, 1):
            x, label = x.to(device), label.to(device)  # 使用GPU
            optimizer.zero_grad()  # Set the gradients to zero (left by previous iteration)
            out = model(x)  # Forward input tensor through your model
            loss = criterion(out, label)  # Calculate loss
            loss.backward()  # Compute gradient of each model parameters base on calculated loss
            optimizer.step()  # Update model parameters using optimizer and gradients

            # Show the training information
            if iteration % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(ep, batch * len(x),
                                                                               len(trainset_loader.dataset),
                                                                               100. * batch / len(trainset_loader),
                                                                               loss.item()))
            iteration += 1

        Val(model)
        # Save trained model
        if ep % 10 == 0:
            state = {'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            torch.save(state, 'checkpoint/p1-%i.pth' % ep)
    # save the final model
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, 'checkpoint/p1-%i.pth' % ep)


def Val(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for x, label in validset_loader:
            x, label = x.to(device), label.to(device)
            out = model(x)
            val_loss += criterion(out, label).item()
            pred = out.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    val_loss /= len(validset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(val_loss, correct,
                                                                                 len(validset_loader.dataset),
                                                                                 100. * correct / len(
                                                                                     validset_loader.dataset)))

if __name__ == '__main__':
    vgg16_bn = Vgg16_bn().to(device)
    training(vgg16_bn, 80)
