import os
import numpy as np
import argparse
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from p3_model import DANN
from p3_dataset import DATA

cudnn.benchmark = True

batch_size = 32
num_iters = 100000
resume_iters = None
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
only = True                                             # 控制是否有target(image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/digits')
    parser.add_argument('--ckp_dir', type=str, default='checkpoint3_2/')
    parser.add_argument('--domain', default='usps', type=str, choices=['mnistm', 'svhn', 'usps'])
    parser.add_argument('--tgt_domain', default='svhn', type=str, choices=['mnistm', 'svhn', 'usps'])
    config = parser.parse_args()
    print(config)

    label_path = os.path.join(config.data_path, config.domain, 'train.csv')
    root = os.path.join(config.data_path, config.domain, 'train')

    label_data = []
    with open(label_path) as f:
        label_data += f.readlines()[1:]

    label_train, label_val = train_test_split(label_data, test_size=0.25, shuffle=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    transform_aug = transforms.Compose([
        transforms.ColorJitter(brightness=63.0 / 255.0, saturation=[0.5, 1.5], contrast=[0.2, 1.8]),
        transforms.Resize([28, 56]),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    trainset = DATA(root=root, label_data=label_train, transform=transform_aug)
    trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    validset = DATA(root=root, label_data=label_val, transform=transform)
    validset_loader = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DANN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005,  betas=[0.9, 0.999])


    def save_checkpoint(step):
        state = {'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        new_checkpoint_path = os.path.join(config.ckp_dir, '{}-dann.pth'.format(step + 1))
        torch.save(state, new_checkpoint_path)
        print('model saved to %s' % new_checkpoint_path)


    def load_checkpoint(resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        new_checkpoint_path = os.path.join(config.ckp_dir, '{}-dann.pth'.format(resume_iters))
        state = torch.load(new_checkpoint_path)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        print('model loaded from %s' % new_checkpoint_path)


    def train(trainset_loader, validset_loader, tgt_trainset_loader=None, config=None):
        criterion = nn.NLLLoss()
        best_acc = 0
        best_loss = 1e15
        iteration = 0
        if resume_iters:
            print("resuming step %d ..." % resume_iters)
            iteration = resume_iters
            load_checkpoint(resume_iters)
            best_loss, best_acc = eval()

        while iteration < num_iters:
            model.train()
            optimizer.zero_grad()

            try:
                data, label = next(data_iter)
            except:
                data_iter = iter(trainset_loader)
                data, label = next(data_iter)

            data, label = data.to(device), label.to(device)
            batch_size = data.size(0)

            p = float(iteration) / (num_iters)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            domain = torch.zeros((batch_size,), dtype=torch.long, device=device)

            class_output, domain_output = model(data, alpha)

            c_loss = criterion(class_output, label)
            d_loss = criterion(domain_output, domain)

            loss = c_loss + d_loss

            if only:
                tgt_d_loss = torch.zeros(1)

            else:
                try:
                    tgt_data, _ = next(tgt_data_iter)
                except:
                    tgt_data_iter = iter(tgt_trainset_loader)
                    tgt_data, _ = next(tgt_data_iter)

                tgt_data = tgt_data.to(device)
                tgt_batch_size = tgt_data.size(0)
                tgt_domain = torch.ones((tgt_batch_size,), dtype=torch.long, device=device)

                _, domain_output = model(tgt_data, alpha)

                tgt_d_loss = criterion(domain_output, tgt_domain)

                loss += tgt_d_loss

            loss.backward()
            optimizer.step()

            # Output training stats
            if (iteration + 1) % 1000 == 0:
                print(
                    'Iteration: {:5d}\tloss: {:.6f}\tloss_class: {:.6f}\tloss_domain: {:.6f}\tloss_tgt_domain: {:.6f}'.format(
                        iteration + 1,loss.item(), c_loss.item(), d_loss.item(), tgt_d_loss.item()))

            # Save model checkpoints
            if (iteration + 1) % 10000 == 0 and iteration > 0:
                val_loss, val_acc = eval()

                save_checkpoint(iteration)

                if (val_acc > best_acc):
                    print('val acc: %.2f > %.2f' % (val_acc, best_acc))
                    best_acc = val_acc
                if (val_loss < best_loss):
                    print('val loss: %.4f < %.4f' % (val_loss, best_loss))
                    best_loss = val_loss

            iteration += 1


    def eval():
        criterion = nn.CrossEntropyLoss()
        model.eval()
        val_loss = 0.0
        correct = 0.0
        with torch.no_grad():
            for data, label in validset_loader:
                data, label = data.to(device), label.to(device)
                output, _ = model(data)
                val_loss += criterion(output, label).item()
                pred = torch.exp(output).max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred)).sum().item()

        val_loss /= len(validset_loader)
        val_acc = 100. * correct / len(validset_loader.dataset)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            val_loss, correct, len(validset_loader.dataset), val_acc))
        return val_loss, val_acc


    if only:
        train(trainset_loader, validset_loader, None, config)

    else:
        tgt_root = os.path.join(config.data_path, config.tgt_domain, 'train')
        tgt_trainset = DATA(root=tgt_root, transform=transform)
        tgt_trainset_loader = DataLoader(tgt_trainset, batch_size=batch_size, shuffle=True, num_workers=1)
        train(trainset_loader, validset_loader, tgt_trainset_loader, config)
