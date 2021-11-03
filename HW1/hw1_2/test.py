import os
import argparse
import glob
import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
from model import Vgg16_FCN32s, Vgg16_FCN8s, Vgg19_FCN32s
from PIL import Image
from shutil import rmtree

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
EVAL = True  # If you wanna use mean_iou_evaluate.py to evaluate

MASK = {
    0: (0, 1, 1),   # (Cyan: 011) Urban land
    1: (1, 1, 0),   # (Yellow: 110) Agriculture land
    2: (1, 0, 1),   # (Purple: 101) Rangeland
    3: (0, 1, 0),   # (Green: 010) Forest land
    4: (0, 0, 1),   # (Blue: 001) Water
    5: (1, 1, 1),   # (White: 111) Barren land
    6: (0, 0, 0),   # (Black: 000) Unknown
}

if __name__ == '__main__':
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='p2_data/validation/')
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/p2-0.pth')
    config = parser.parse_args()
    print(config)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    # Get device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('Device used:', device)

    # Load model
    #model = Vgg16_FCN32s().to(device)
    #model = Vgg16_FCN8s().to(device)
    model = Vgg19_FCN32s().to(device)
    state = torch.load(config.checkpoint_path)
    model.load_state_dict(state['state_dict'])

    # save directory
    filenames = glob.glob(os.path.join(config.img_dir, '*.jpg'))
    filenames = sorted(filenames)

    model.eval()
    with torch.no_grad():
        for name in filenames:
            ImageID = name.split('/')[-1].split('_')[0]
            output_filename = os.path.join(config.output_path, '{}_mask.png'.format(ImageID))
            x = Image.open(name)
            x = val_transform(x)
            data_shape = x.shape
            x = torch.unsqueeze(x, 0)
            x = x.to(device)
            output = model(x)
            pred = output.max(1, keepdim=True)[1].reshape((-1, data_shape[1], data_shape[2]))  # 給出預測矩陣中最大值的位置
            y = torch.zeros((pred.shape[0], 3, pred.shape[1], pred.shape[2]))
            # 加入顏色
            for k, v in MASK.items():
                y[:, 0, :, :][pred == k] = v[0]
                y[:, 1, :, :][pred == k] = v[1]
                y[:, 2, :, :][pred == k] = v[2]

            y = transforms.ToPILImage()(y.squeeze())
            y.save(output_filename)

    # Evaluate prediction results
    if EVAL:
        from mean_iou_evaluate import read_masks, mean_iou_score
        pred = read_masks(config.output_path)
        labels = read_masks('p2_data/validation/')
        mean_iou_score(pred, labels)
