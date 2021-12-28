import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image

filenameToPILImage = lambda x: Image.open(x)


# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)


class NShotTaskSampler(Sampler):
    def __init__(self, csv_path, episodes_per_epoch, N_way, N_shot, N_query):
        self.data_df = pd.read_csv(csv_path)
        self.N_way = N_way
        self.N_shot = N_shot
        self.N_query = N_query
        self.episodes_per_epoch = episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            episode_classes = np.random.choice(self.data_df['label'].unique(), size=self.N_way, replace=False)

            support = []
            query = []

            for k in episode_classes:
                ind = self.data_df[self.data_df['label'] == k]['id'].sample(self.N_shot + self.N_query).values
                support = support + list(ind[:self.N_shot])
                query = query + list(ind[self.N_shot:])

            batch = support + query

            yield np.stack(batch)

    def __len__(self):
        return self.episodes_per_epoch


class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence)

    def __len__(self):
        return len(self.sampled_sequence)
