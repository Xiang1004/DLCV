import glob
import os
from torch.utils.data import Dataset
from PIL import Image

Num_Class = 37


class DATA(Dataset):
    def __init__(self, root, transform=None):
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        for i in range(Num_Class):
            filenames = glob.glob(os.path.join(root, str(i) + '_' + '*.jpg'))
            for name in filenames:
                self.filenames.append((name, i))  # (filename, label)
        self.len = len(self.filenames)

    def __getitem__(self, index):
        image_name, label = self.filenames[index]
        image = Image.open(image_name).convert('RGB')
        image = self.transform(image)
        return image, label

    def __len__(self):
        return self.len
