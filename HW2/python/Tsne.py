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
# TSNE
import pandas as pd
import math
from sklearn.manifold import TSNE

Tsne = True

def main(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([28, 56]),
        transforms.CenterCrop(28),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DANN().to(device)

    state = torch.load(config.ckp_dir)
    model.load_state_dict(state['state_dict'])

    filenames = glob.glob(os.path.join(config.tgt_dir, '*.png'))
    filenames = sorted(filenames)

    out_filename = config.output_dir
    os.makedirs(os.path.dirname(config.output_dir), exist_ok=True)

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

    # TSNE
    if Tsne:
        batch_size = 1024

        class labelImgData(Dataset):
            def __init__(self, root, transform=None):
                self.root = root
                filenames = sorted(glob.glob(os.path.join(self.root, '*.png')))
                self.trans = transform
                self.len = len(filenames)
                fn = root + ".csv"
                labels = np.array(pd.read_csv(fn)['label'])
                self.fnLabelList = [(filenames[i], labels[i]) for i in range(self.len)]
                self.discardedfnLabelList = []

            def __getitem__(self, index):
                fn, label = self.fnLabelList[index]
                img = Image.open(fn)
                img = img.convert('RGB')
                img = img if self.trans == None else self.trans(img)
                return img, label

            def rngDiscarding(self, reserveCnt=5000):
                reserveCnt = min(reserveCnt, self.len)
                self.fnLabelList += self.discardedfnLabelList
                random.shuffle(self.fnLabelList)
                self.discardedfnLabelList = self.fnLabelList[reserveCnt:]
                self.fnLabelList = self.fnLabelList[:reserveCnt]
                self.len = len(self.fnLabelList)

            def restoreFromDiscarding(self):
                self.fnLabelList += self.discardedfnLabelList
                self.discardedfnLabelList = []
                self.len = len(self.fnLabelList)

            def __len__(self):
                return self.len

        img_transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

        mnist_test = labelImgData(root='data/digits/mnistm/test', transform=img_transform)
        svhn_test = labelImgData(root='data/digits/svhn/test', transform=img_transform)
        usps_test = labelImgData(root='data/digits/usps/test', transform=img_transform)

        mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)
        svhn_test_loader = DataLoader(svhn_test, batch_size=batch_size, shuffle=False, num_workers=0)
        usps_test_loader = DataLoader(usps_test, batch_size=batch_size, shuffle=False, num_workers=0)

        def getLatentVectorList(src_loader, tgt_loader):
            model.eval()
            with torch.no_grad():
                img_src, label_src = iter(src_loader).next()
                img_tgt, label_tgt = iter(tgt_loader).next()

                ratio = math.sqrt(len(src_loader.dataset) / len(tgt_loader.dataset))
                if ratio < 1:
                    length = int(ratio * len(img_tgt))
                    img_src, label_src = img_src[:length], label_src[:length]
                else:
                    length = int(len(img_src) / ratio)
                    img_tgt, label_tgt = img_tgt[:length], label_tgt[:length]

                feature = model.cnn_layers(img_src.to(device))
                feature = feature.view(-1, 50 * 4 * 4)
                latentVectorList = [feat.cpu().numpy() for feat in feature]

                feature = model.cnn_layers(img_tgt.to(device))
                feature = feature.view(-1, 50 * 4 * 4)
                latentVectorList += [feat.cpu().numpy() for feat in feature]

                labelList = label_src.cpu().numpy().tolist() + label_tgt.cpu().numpy().tolist()
                domainList = np.ones(len(img_src)).tolist() + np.zeros(len(img_tgt)).tolist()
            return np.array(latentVectorList), np.array(labelList), np.array(domainList)

        def plotLabelingTSNE(args, ax):
            latentVectorList, labelList, domainList = args
            latentVectorList = TSNE(init='random', random_state=5).fit_transform(latentVectorList)
            cm = plt.get_cmap('gist_rainbow')
            for label in range(10):
                color = cm(1. * label / 10)
                xy = np.array([vec for i, vec in enumerate(latentVectorList) if label == labelList[i]])
                ax[0].scatter(xy[:, 0], xy[:, 1], s=8, color=color, label=str(label), alpha=0.9)
                ax[0].legend()
            for domain in range(2):
                color, gender = ("cornflowerblue", "source domain") if domain else ("hotpink", "target domain")
                xy = np.array([vec for i, vec in enumerate(latentVectorList) if domain == domainList[i]])
                ax[1].scatter(xy[:, 0], xy[:, 1], s=8, color=color, label=gender, alpha=0.9)
                ax[1].legend()

        fig, ax = plt.subplots(3, 2, figsize=(15, 18))
        srcStrList, srcLoadList = ['mnistm','usps', 'svhn'], [mnist_test_loader, usps_test_loader, svhn_test_loader]
        tgtStrList, tgtLoadList = ['usps', 'svhn', 'mnistm'], [usps_test_loader, svhn_test_loader,mnist_test_loader]
        stringLoaderTupleList = zip(srcStrList, srcLoadList, tgtStrList, tgtLoadList)

        for i, args in enumerate(stringLoaderTupleList):
            srcStr, srcLoad, tgtStr, tgtLoad = args

            plotLabelingTSNE(getLatentVectorList(srcLoad, tgtLoad), ax[i])

            ax[i][0].set_title(srcStr + '->' + tgtStr + ' classes (0-9)')
            ax[i][1].set_title(srcStr + '->' + tgtStr + ' domains (source/target)')
        plt.savefig('p3_T-SNE.jpg')

    # Accuracy

    with open(os.path.join(config.output_dir), mode='r') as predict:
        reader = csv.reader(predict)
        pred_dict = {rows[0]: rows[1] for rows in reader}

    with open(os.path.join(config.tgt_dir + '.csv'), mode='r') as gt:
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
    parser.add_argument('--tgt_dir', type=str, default='data/digits/svhn/test')
    parser.add_argument('--output_dir', type=str, default='checkpoint/pred.csv')
    parser.add_argument('--ckp_dir', type=str, default='checkpoint/p4_3-dann.pth')

    config = parser.parse_args()
    print(config)
    main(config)
