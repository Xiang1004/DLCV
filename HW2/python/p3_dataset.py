import os
import glob
from torch.utils.data import Dataset
from PIL import Image


class DATA(Dataset):
    def __init__(self, root, label_data=None, transform=None):
        self.transform = transform
        self.label_data = label_data
        self.filenames = []
        self.labels = []

        if label_data is not None:
            for fn in self.label_data:
                img_path, label = fn.split(',')
                self.filenames.append(os.path.join(root, img_path))
                self.labels.append(int(label))
        else:
            for fn in glob.glob(os.path.join(root, '*.png')):
                self.filenames.append(fn)
                self.labels.append(0)

        self.len = len(self.filenames)

    def __getitem__(self, index):
        image_name, label_class = self.filenames[index], self.labels[index]
        image = Image.open(image_name).convert('RGB')
        image = self.transform(image)
        return image, label_class

    def __len__(self):
        return self.len
