import os
import argparse
import glob
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from p1_model import ViT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--checkpoint_path', type=str, default='p1.pth')
    config = parser.parse_args()
    print(config)

    val_transform = transforms.Compose([
        transforms.Resize((255)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    model = ViT().to(device)
    state = torch.load(config.checkpoint_path)
    model.load_state_dict(state['state_dict'])
    model.eval()

    filenames = glob.glob(os.path.join(config.img_path, '*.jpg'))
    filenames = sorted(filenames)

    with open(os.path.join(config.output_path), 'w+') as f:
        correct = 0  # label
        f.write('image_id,label\n')
        with torch.no_grad():
            for name in filenames:
                x = Image.open(name).convert('RGB')
                x = val_transform(x)
                x = torch.unsqueeze(x, 0)
                x = x.to(device)
                y = model(x)
                pred = y.max(1, keepdim=True)[1]  # get the index of the max log-probability
                f.write(name.split('/')[-1] + ',' + str(pred.item()) + '\n')

                if pred.item() == int(name.split('/')[-1].split('_')[0]):  # 判斷正確率
                    correct += 1

    print("Acurracy = " + str(correct / len(filenames)))
