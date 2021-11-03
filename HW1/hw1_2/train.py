import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Vgg16_FCN32s, Vgg16_FCN8s, Vgg19_FCN32s
from dataset import DATA

BATCH = 1   # pic by pic
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

# Load dataset
trainset = DATA(root='p2_data/train/', transform=train_transform)
trainset_loader = DataLoader(trainset, batch_size=BATCH, shuffle=True,  num_workers=0)

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(100)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

def train_fcn32s(model, epoch, log_interval):
    model.train()  # set training mode
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    iteration = 0
    for ep in range(epoch):
        for batch, (x, label) in enumerate(trainset_loader):
            x, label = x.to(device), label.to(device)
            output = model(x)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(ep, batch * len(x), len(trainset_loader.dataset),
                                                                               100. * batch / len(trainset_loader), loss.item()))
            iteration += 1
        
        if ep % 10 == 0:
            state = {'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            torch.save(state, 'checkpoint/p2-%i.pth' % ep)
    # save the final model
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, 'p2-%i.pth' % ep)

if __name__ == '__main__':
    #fcn32s = Vgg16_FCN32s().to(device)
    vgg19_fcn32s = Vgg19_FCN32s().to(device)
    train_fcn32s(vgg19_fcn32s, 100, log_interval=200)


