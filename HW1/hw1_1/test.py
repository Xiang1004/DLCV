import os
import argparse
import glob
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from model import Vgg16_bn

INPUT_SIZE = 224
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
TSNE = True

if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='p1_data/val_50/')
    parser.add_argument('--output_path', type=str, default='p1_data/')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/p1-70.pth')
    config = parser.parse_args()
    print(config)

    val_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)

    # Load model
    vgg16_bn = Vgg16_bn().to(device)
    state = torch.load(config.checkpoint_path)
    vgg16_bn.load_state_dict(state['state_dict'])
    vgg16_bn.eval()

    filenames = glob.glob(os.path.join(config.img_dir, '*.png'))
    filenames = sorted(filenames)

    with open(os.path.join(config.output_path, 'pred.csv'), 'w') as f:
        correct = 0  # label
        f.write('image_id,label\n')
        with torch.no_grad():
            # vgg16_bn[3](倒數第二層)的output (需在呼叫vgg16之前)
            if TSNE:
                feature_dic = {}
                def get_activation(name):
                    def hook(model, input, output):
                        feature_dic[name] = output.detach()
                    return hook
                vgg16_bn.vgg16.classifier[3].register_forward_hook(get_activation('features'))
                feature_array = []
                label_array = []

            for name in filenames:
                x = Image.open(name)
                x = val_transform(x)
                x = torch.unsqueeze(x, 0)     # 增加維度
                x = x.to(device)
                y = vgg16_bn(x)
                pred = y.max(1, keepdim=True)[1]  # get the index of the max log-probability
                f.write(name.split('/')[-1] + ',' + str(pred.item()) + '\n')
                if TSNE:
                    feature_array.append(feature_dic['features'].to("cpu").numpy()[0])
                    label_array.append(int(name.split('/')[-1].split('_')[0]))

                if pred.item() == int(name.split('/')[-1].split('_')[0]):  # 判斷正確率
                    correct += 1

    feature_array = np.array(feature_array)
    label_array = np.array(label_array)

    print("Acurracy = " + str(correct / len(filenames)))

    if TSNE:
        from sklearn import manifold

        # Get random color set
        color_dict = {}
        for i in range(50):
            color = []
            for c in range(3):
                color.append(random.random())
            color.append(1)
            color_dict[i] = tuple(color)

        # t-SNE
        X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(feature_array)

        # Data Visualization
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # Normalize
        plt.figure(figsize=(10, 10))
        for i in range(X_norm.shape[0]):
            color_num = color_dict[label_array[i]]
            plt.text(X_norm[i, 0], X_norm[i, 1], str(label_array[i]), color=color_num,
                     fontdict={'size': 8})
        plt.xticks([])
        plt.yticks([])
        plt.savefig('t-SNE.jpg')
