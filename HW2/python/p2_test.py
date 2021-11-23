import argparse
import os
import sys
import torch
import random
from p2_model import Generator
from torchvision.utils import save_image
import PIL.Image as Image

image_100 = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

manualSeed = 1004
random.seed(manualSeed)
torch.manual_seed(manualSeed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--ckp_path', type=str, default='model/p2.pth')

    config = parser.parse_args()

    batchsize = 1

    G = Generator().to(device)
    G.load_state_dict(torch.load(config.ckp_path, map_location=device))
    G.eval()

    with torch.no_grad():
        for i in range(10):
            for j in range(100):
                noise = torch.randn(batchsize, 100, device=device)
                num_int = [i]
                num_tensor = torch.tensor(num_int).to(device)

                fake_images = G(noise, num_tensor)
                save_image(fake_images, os.path.join(config.save_path, '{}_{:03}.png'.format(i, j+1)), nrow=1)

        if image_100:
            to_image = Image.new('RGB', (280, 280))
            for row in range(10):
                for column in range(10):
                    pic_fname = f"{column}_{row + 1:03}.png"
                    from_image = Image.open(os.path.join(config.save_path, pic_fname))
                    to_image.paste(from_image, (column * 28, row * 28))
            to_image.save("p2.png")

