import os
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
from p1_model import Generator
from PIL import Image
import random

image_32 = False

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str)
parser.add_argument('--ckp_path', type=str, default='model/p1.pth')
config = parser.parse_args()
print(config)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Generator().to(device)

state = torch.load(config.ckp_path, map_location=device)
model.load_state_dict(state)
model.eval()

seed = 1004
torch.manual_seed(seed)
random.seed(seed)
with torch.no_grad():
    for i in range(1000):
        z = torch.randn((1, 128, 1, 1)).to(device)
        predict = model(z)
        torchvision.utils.save_image(predict, config.save_path + '{0:0=4d}.png'.format(i+1), nrow=8, normalize=True)

    if image_32:
        i = 0
        to_image = Image.new('RGB', (512, 256))
        for row in range(4):
            for column in range(8):
                image = Image.open(config.save_path + '{0:0=4d}.png'.format(i+1))
                to_image.paste(image, (column * 64, row * 64))
                i = i + 1
        to_image.save("p1.png")