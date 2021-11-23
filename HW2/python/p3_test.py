import os
import argparse
import glob
import csv, sys
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
from p3_model import DANN
from torch.utils.data import DataLoader, Dataset

def main(config):
    if config.target_domain == 'mnistm':
        path = config.ckp_path1
    elif config.target_domain == 'usps':
        path = config.ckp_path2
    elif config.target_domain == 'svhn':
        path = config.ckp_path3
    else:
        print('wrong target domain !')

    if path == 'model/p3_3.pth':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([28, 56]),
            transforms.CenterCrop(28),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DANN().to(device)

    state = torch.load(path, map_location=device)
    model.load_state_dict(state['state_dict'])

    filenames = glob.glob(os.path.join(config.img_path, '*.png'))
    filenames = sorted(filenames)

    out_filename = config.save_path
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    model.eval()
    with open(out_filename, 'w') as f:
        f.write('image_name,label\n')
        with torch.no_grad():
            for name in filenames:
                x = Image.open(name).convert('RGB')
                x = transform(x)
                x = torch.unsqueeze(x, 0)
                x = x.to(device)
                y, _ = model(x, 1)
                pred = y.max(1, keepdim=True)[1]  # get the index of the max log-probability
                f.write(name.split('/')[-1] + ',' + str(pred.item()) + '\n')

    # Accuracy

    with open(os.path.join(config.save_path), mode='r') as predict:
        reader = csv.reader(predict)
        pred_dict = {rows[0]: rows[1] for rows in reader}

    with open(os.path.join(config.img_path + '.csv'), mode='r') as gt:
        reader = csv.reader(gt)
        gt_dict = {rows[0]: rows[1] for rows in reader}

    total_count = 0
    correct_count = 0
    for key, value in pred_dict.items():
        if key not in gt_dict:
            sys.exit("Item mismatch: \"{}\" does not exist in the provided ground truth file.".format(key))
        if value == 'label':
            continue
        if gt_dict[key] == value:
            correct_count += 1
        total_count += 1

    accuracy = (correct_count / total_count) * 100
    print('Accuracy: {}/{} ({}%)'.format(correct_count, total_count, accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--target_domain', type=str)
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--ckp_path1', type=str, default='model/p3_1.pth')
    parser.add_argument('--ckp_path2', type=str, default='model/p3_2.pth')
    parser.add_argument('--ckp_path3', type=str, default='model/p3_3.pth')

    config = parser.parse_args()
    print(config)
    main(config)
