import glob
import os
from torch.utils.data import Dataset
from PIL import Image

Num_Class = 50


class DATA(Dataset):
    def __init__(self, root, transform=None):
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        for i in range(Num_Class):
            filenames = glob.glob(os.path.join(root, str(i) + '_' + '*.png'))
            for name in filenames:
                self.filenames.append((name, i))  # (filename, label)
        self.len = len(self.filenames)

    def __getitem__(self, index):
        image_name, label = self.filenames[index]
        image = Image.open(image_name)
        # Do some processing on the photo (e.g. Resize、Normalize...)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
