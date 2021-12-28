import os
import argparse
import glob
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from p2_model import SSL
import csv

import sys
import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

dict = {
    "Couch":        0,
    "Helmet":       1,
    "Refrigerator": 2,
    "Alarm_Clock":  3,
    "Bike":         4,
    "Bottle":       5,
    "Calculator":   6,
    "Chair":        7,
    "Mouse":        8,
    "Monitor":      9,
    "Table":        10,
    "Pen":          11,
    "Pencil":       12,
    "Flowers":      13,
    "Shelf":        14,
    "Laptop":       15,
    "Speaker":      16,
    "Sneakers":     17,
    "Printer":      18,
    "Calendar":     19,
    "Bed":          20,
    "Knives":       21,
    "Backpack":     22,
    "Paper_Clip":   23,
    "Candles":      24,
    "Soda":         25,
    "Clipboards":   26,
    "Fork":         27,
    "Exit_Sign":    28,
    "Lamp_Shade":   29,
    "Trash_Can":    30,
    "Computer":     31,
    "Scissors":     32,
    "Webcam":       33,
    "Sink":         34,
    "Postit_Notes": 35,
    "Glasses":      36,
    "File_Cabinet": 37,
    "Radio":        38,
    "Bucket":       39,
    "Drill":        40,
    "Desk_Lamp":    41,
    "Toys":         42,
    "Keyboard":     43,
    "Notebook":     44,
    "Ruler":        45,
    "ToothBrush":   46,
    "Mop":          47,
    "Flipflops":    48,
    "Oven":         49,
    "TV":           50,
    "Eraser":       51,
    "Telephone":    52,
    "Kettle":       53,
    "Curtains":     54,
    "Mug":          55,
    "Fan":          56,
    "Push_Pin":     57,
    "Batteries":    58,
    "Pan":          59,
    "Marker":       60,
    "Spoon":        61,
    "Screwdriver":  62,
    "Hammer":       63,
    "Folder":       64
    }
new_dict = {v : k for k, v in dict.items()}

def DATA(csv_path):
    csv_path = os.path.join(csv_path)
    lines = [x.strip() for x in open(csv_path, 'r').readlines()][1: ]

    for l in lines:
        name = l.split(',')[1]
        label = dict[name.split('0')[0]]
    return name, label



if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_csv_path', type=str)
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--checkpoint_path', type=str, default='./p2.pth')
    config = parser.parse_args()
    print(config)

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device used:', device)

    # Load model
    resnet50 = models.resnet50(pretrained=False).to(device)
    resnet50.load_state_dict(torch.load('./resnet_pretrain.pth'))
    resnet50.eval()

    model = SSL().to(device)
    state = torch.load(config.checkpoint_path)
    model.load_state_dict(state['state_dict'])
    model.eval()

    filenames = glob.glob(os.path.join(config.img_path, '*.jpg'))
    filenames = sorted(filenames)

    with open(os.path.join(config.output_path), 'w+') as f:
        correct = 0  # label
        f.write('id,filename,label\n')
        with torch.no_grad():
            i = 0
            for name in filenames:
                x = Image.open(name)
                x = val_transform(x)
                x = torch.unsqueeze(x, 0)     # 增加維度
                x = x.to(device)
                resnet = resnet50(x).to(device)
                y = model(resnet)
                pred = y.max(1, keepdim=True)[1]  # get the index of the max log-probability

                f.write(str(i) + ',' + name.split('/')[-1] + ',' + str(new_dict[pred.item()]) + '\n')
                i += 1

    with open(os.path.join(config.output_path), mode='r') as predict:
        reader = csv.reader(predict)
        pred_dict = {rows[1]: rows[2] for rows in reader}

    with open(os.path.join(config.img_csv_path), mode='r') as gt:
        reader = csv.reader(gt)
        gt_dict = {rows[1]: rows[2] for rows in reader}

    total_count = 0
    correct_count = 0
    for key, value in pred_dict.items():
        if key not in gt_dict:
            sys.exit("Item mismatch: \"{}\" does not exist in the provided ground truth file.".format(key))
        if value == 'label':
            continue
        if dict[str(gt_dict[key])] == dict[str(value)]:
            correct_count += 1
        total_count += 1

    accuracy = (correct_count / total_count) * 100
    print('Accuracy: {}/{} ({}%)'.format(correct_count, total_count, accuracy))

