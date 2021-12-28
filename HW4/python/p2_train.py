import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from p2_model import SSL
from p2_dataset import DATA
from train import SelfSupervisedLearner

Batch = 32

trainset = DATA('./p2_data/office/train.csv', './p2_data/office/train')
validset = DATA('./p2_data/office/val.csv', './p2_data/office/val')
trainset_loader = DataLoader(trainset, batch_size=Batch, shuffle=True, num_workers=0)
validset_loader = DataLoader(validset, batch_size=Batch, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device used:', device)

def training(model, epoch):
    optimizer = optim.SGD(list(model.parameters()) + list(resnet50.parameters()), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()  # set training model

    iteration = 0
    for ep in range(epoch):
        for batch, (x, label) in enumerate(trainset_loader, 1):

            x, label = x.to(device), label.to(device)  # 使用GPU

            optimizer.zero_grad()  # Set the gradients to zero (left by previous iteration)

            resnet = resnet50(x).to(device)
            out = model(resnet)  # Forward input tensor through your model

            loss = criterion(out, label)  # Calculate loss
            loss.backward()  # Compute gradient of each model parameters base on calculated loss

            optimizer.step()
            # Show the training information
            if iteration % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(ep, batch * len(x),
                                                                               len(trainset_loader.dataset),
                                                                               100. * batch / len(trainset_loader),
                                                                               loss.item()))
            iteration += 1

        Val(model)
        # Save trained model
        if ep % 1 == 0:
            state = {'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
            torch.save(state, 'checkpoint/p2-%i.pth' % ep)
    # save the final model
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, 'checkpoint/p2-%i.pth' % ep)


def Val(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for x, label in validset_loader:

            x, label = x.to(device), label.to(device)
            resnet = resnet50(x).to(device)
            out = model(resnet)
            val_loss += criterion(out, label).item()
            pred = out.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    val_loss /= len(validset_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(val_loss, correct, len(validset_loader.dataset), 100. * correct / len(validset_loader.dataset)))


if __name__ == '__main__':

    resnet50 = models.resnet50(pretrained=True).to(device)
    resnet50.load_state_dict(torch.load('./checkpoint/resnet_pretrain.pth'))



    ssl = SSL().to(device)
    training(ssl, 100)
