import glob
import os
from torch.utils.data import Dataset
from PIL import Image

class DATA(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.filename = []

        for fn in glob.glob(os.path.join(root, '*.png')):
            self.filename.append(fn)

        self.len = len(self.filename)


    def __getitem__(self, index):
        """ Get a sample from the dataset """
        fn = self.filename[index]
        img = Image.open(fn)
        if self.transform is not None:
            img = self.transform(img)

        return img, img

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len