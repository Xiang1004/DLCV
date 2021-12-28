import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import csv
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR

from p1_dataset import MiniDataset, GeneratorSampler, NShotTaskSampler
from p1_model import MLP

epochs = 100

if __name__ == '__main__':

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    SEED = 123
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    # training configuration.
    parser.add_argument('--episodes_per_epoch', default=600, type=int)
    parser.add_argument('--N_way_train', default=5, type=int)
    parser.add_argument('--N_shot_train', default=1, type=int)
    parser.add_argument('--N_query_train', default=15, type=int)
    parser.add_argument('--N_way_val', default=5, type=int)
    parser.add_argument('--N_shot_val', default=1, type=int)
    parser.add_argument('--N_query_val', default=15, type=int)
    parser.add_argument('--matching_fn', default='l2', type=str, choices=['l2', 'cosine', 'parametric'])
    # path.
    parser.add_argument('--train_csv', type=str, default='./p1_data/mini/train.csv')
    parser.add_argument('--train_data_path', type=str, default='./p1_data/mini/train')
    parser.add_argument('--val_csv', type=str, default='./p1_data/mini/val.csv')
    parser.add_argument('--val_data_path', type=str, default='./p1_data/mini/val')
    parser.add_argument('--val_testcase_csv', type=str, default='./p1_data/mini/val_testcase.csv')
    parser.add_argument('--ckp_path', default='./checkpoint/', type=str)

    config = parser.parse_args()
    print(config)

    # Dataloader
    train_dataset = MiniDataset(config.train_csv, config.train_data_path)
    val_dataset = MiniDataset(config.val_csv, config.val_data_path)

    train_loader = DataLoader(train_dataset, num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
                              batch_sampler=NShotTaskSampler(config.train_csv, config.episodes_per_epoch,
                                                             config.N_way_train, config.N_shot_train,
                                                             config.N_query_train))

    val_loader = DataLoader(val_dataset, batch_size=config.N_way_val * (config.N_query_val + config.N_shot_val),
                            num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
                            sampler=GeneratorSampler(config.val_testcase_csv))

    # build_model
    model = MLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)

    if config.matching_fn == 'parametric':
        parametric = nn.Sequential(
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(400, 1)
        ).to(sdevice)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(parametric.parameters()),
                                      lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)

    scheduler = StepLR(optimizer, step_size=40, gamma=0.9)

    def save_checkpoint(step):
        state = {'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        if config.matching_fn == 'parametric':
            state = {'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'parametric': parametric.state_dict()}

        save_path = os.path.join(config.ckp_path, '{}-protonet.pth'.format(step + 1))
        torch.save(state, save_path)
        print('Model Saved!')

    def pairwise_distances(x, y, matching_fn=, parametric=None):
        n_x = x.shape[0]
        n_y = y.shape[0]

        if matching_fn == 'l2':
            distances = (
                    x.unsqueeze(1).expand(n_x, n_y, -1) -
                    y.unsqueeze(0).expand(n_x, n_y, -1)
            ).pow(2).sum(dim=2)
            return distances

        elif matching_fn == 'cosine':
            cos = nn.CosineSimilarity(dim=2, eps=1e-6)
            cosine_similarities = cos(x.unsqueeze(1).expand(n_x, n_y, -1), y.unsqueeze(0).expand(n_x, n_y, -1))
            return 1 - cosine_similarities

        elif matching_fn == 'parametric':
            x_exp = x.unsqueeze(1).expand(n_x, n_y, -1).reshape(n_x * n_y, -1)
            y_exp = y.unsqueeze(0).expand(n_x, n_y, -1).reshape(n_x * n_y, -1)
            distances = parametric(torch.cat([x_exp, y_exp], dim=-1))
            return distances.reshape(n_x, n_y)

        else:
            raise (ValueError('Unsupported similarity function'))

    def train():
        criterion = nn.CrossEntropyLoss()

        best_mean = 0
        iteration = 0
        episodic_acc = []

        for ep in range(epochs):
            model.train()

            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()

                support_input = data[:config.N_way_train * config.N_shot_train, :, :, :]
                query_input = data[config.N_way_train * config.N_shot_train:, :, :, :]

                label_encoder = {target[i * config.N_shot_train]: i for i in range(config.N_way_train)}
                query_label = torch.cuda.LongTensor([label_encoder[class_name]
                                                     for class_name in target[config.N_way_train * config.N_shot_train:]])

                support = model(support_input)
                queries = model(query_input)
                prototypes = support.reshape(config.N_way_train, config.N_shot_train, -1).mean(dim=1)

                if config.matching_fn == 'parametric':
                    distances = pairwise_distances(queries, prototypes, config.matching_fn, parametric)

                else:
                    distances = pairwise_distances(queries, prototypes, config.matching_fn)

                loss = criterion(-distances, query_label)
                loss.backward()
                optimizer.step()

                y_pred = (-distances).softmax(dim=1).max(1, keepdim=True)[1]
                episodic_acc.append(1. * y_pred.eq(query_label.view_as(y_pred)).sum().item() / len(query_label))

                if (iteration + 1) % 300 == 0:
                    episodic_acc = np.array(episodic_acc)
                    mean = episodic_acc.mean()
                    std = episodic_acc.std()

                    print('Epoch: {:3d} [{:d}/{:d}]\tIteration: {:5d}\tLoss: {:.6f}\tAccuracy: {:.2f} +- {:.2f} %'
                          .format(ep, (batch_idx + 1), len(train_loader), iteration + 1, loss.item(),
                                  mean * 100, 1.96 * std / 300**(1/2) * 100))

                    episodic_acc = []

                if (iteration + 1) % 600 == 0:
                    loss, mean, std = val()
                    if mean > best_mean:
                        best_mean = mean
                        save_checkpoint(iteration)

                iteration += 1

            scheduler.step()

    def val():
        criterion = nn.CrossEntropyLoss()
        model.eval()
        episodic_acc = []
        loss = []

        with torch.no_grad():
            for b_idx, (data, target) in enumerate(val_loader):
                data = data.to(device)
                support_input = data[:config.N_way_val * config.N_shot_val, :, :, :]
                query_input = data[config.N_way_val * config.N_shot_val:, :, :, :]

                label_encoder = {target[i * config.N_shot_val]: i for i in range(config.N_way_val)}
                query_label = torch.cuda.LongTensor([label_encoder[class_name]
                                                     for class_name in target[config.N_way_val * config.N_shot_val:]])

                support = model(support_input)
                queries = model(query_input)
                prototypes = support.reshape(config.N_way_val, config.N_shot_val, -1).mean(dim=1)

                if config.matching_fn == 'parametric':
                    distances = pairwise_distances(queries, prototypes, config.matching_fn, parametric)
                else:
                    distances = pairwise_distances(queries, prototypes, config.matching_fn)

                loss.append(criterion(-distances, query_label).item())
                y_pred = (-distances).softmax(dim=1).max(1, keepdim=True)[1]
                episodic_acc.append(1. * y_pred.eq(query_label.view_as(y_pred)).sum().item() / len(query_label))

        loss = np.array(loss)
        episodic_acc = np.array(episodic_acc)
        loss = loss.mean()
        mean = episodic_acc.mean()
        std = episodic_acc.std()

        print('\nLoss: {:.6f}\tAccuracy: {:.2f} +- {:.2f} %\n'.format(loss, mean * 100, 1.96 * std / 600**(1/2) * 100))

        return loss, mean, std

    train()
