import glob
import os
import torch
import copy
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from torch.utils.data import Dataset
from PIL import Image

N_CLASS = 7

def viz_mask(image):
    image = transforms.ToTensor()(image)
    image = 4 * image[0] + 2 * image[1] + 1 * image[2]
    mask = torch.zeros(image.shape, dtype=torch.long)
    mask[image == 3] = 0
    mask[image == 6] = 1
    mask[image == 5] = 2
    mask[image == 2] = 3
    mask[image == 1] = 4
    mask[image == 7] = 5
    mask[image == 0] = 6
    return mask

class DATA(Dataset):
    def __init__(self, root, transform=None):
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform
        self.sate_img = []
        self.mask_img = []

        # read filenames
        sate_filenames = glob.glob(os.path.join(root, '*.jpg'))
        # mask_filenames = glob.glob(os.path.join(root, '*.png'))
        for name in sate_filenames:
            self.filenames.append((name, name[:-7] + 'mask.png'))  # (sate_filename, mask_filename) pair
        self.len = len(self.filenames)

        # Load image to memory
        for name in self.filenames:
            self.sate_img.append(copy.deepcopy(Image.open(name[0])))
            self.mask_img.append(copy.deepcopy(Image.open(name[1])))

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        # from memory 
        sate_img = self.sate_img[index]
        mask_img = self.mask_img[index]

        if random.random() > 0.5:
            sate_img = TF.hflip(sate_img)
            mask_img = TF.hflip(mask_img)

        if random.random() > 0.5:
            sate_img = TF.vflip(sate_img)
            mask_img = TF.vflip(mask_img)

        if self.transform is not None:
            sate_img = self.transform(sate_img)

        return sate_img, viz_mask(mask_img)

    def __len__(self):
        return self.len
