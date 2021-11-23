import os
import csv
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class DATA(Dataset):
    def __init__(self, root, transform=None):
        self.labels = []
        self.transform = transform

        with open(os.path.join(root, 'train.csv'), newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                train_pic = os.path.join(root, 'train', row['image_name'])
                label = int(row['label'])
                ar_label = np.zeros(10)
                ar_label[label] = 1
                self.labels.append((train_pic, label))
        self.len = len(self.labels)

    def __getitem__(self, item):
        image_name, label_class = self.labels[item]
        image = Image.open(image_name)
        image = self.transform(image)
        return image, label_class

    def __len__(self):
        return self.len