import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from p1_model import Generator, Discriminator
from p1_dataset import DATA

batch_size = 24
epoch = 80

manualSeed = 1004
random.seed(manualSeed)
torch.manual_seed(manualSeed)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    trainset = DATA(root='data/face/train', transform=transform)
    trainsetloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #build_model():
    G = Generator().to(device)
    D = Discriminator().to(device)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def save_checkpoint(step):
        G_path = os.path.join('checkpoint/', '{}-G.pth'.format(ep))
        D_path = os.path.join('checkpoint/', '{}-D.pth'.format(ep))
        torch.save(G.state_dict(), G_path)
        torch.save(D.state_dict(), D_path)
        print('Saved model checkpoints into {}...'.format('checkpoint/'))

    for ep in range(epoch):
        for batch, (x, _) in enumerate(trainsetloader, 1):

            ################
            #   train D    #
            ################
            noise = torch.randn(batch_size, 128, 1, 1, device=device)
            fake = G(noise)

            Timage = x.to(device)
            Tp = D(Timage)
            Ttarget = torch.ones(Timage.size(0), 1, device=device)
            Tloss = F.binary_cross_entropy(Tp, Ttarget)

            Fp = D(fake)
            Ftarget = torch.zeros(fake.size(0), 1, device=device)
            Floss = F.binary_cross_entropy(Fp, Ftarget)

            D_loss = Tloss + Floss
            d_optimizer.zero_grad()
            D_loss.backward()
            d_optimizer.step()

            ################
            #   train G    #
            ################
            noise = torch.randn(batch_size, 128, 1, 1, device=device)
            fake = G(noise)

            pred = D(fake)
            target = torch.ones(batch_size, 1, device=device)

            G_loss = F.binary_cross_entropy(pred, target)
            g_optimizer.zero_grad()
            G_loss.backward()
            g_optimizer.step()

            # Output training states

            if batch % 100 == 0 or batch == len(trainsetloader):
                print('Epoch {} Iteration {}: Discriminator_loss {:.3f} Generator_loss {:.3f}'.format(ep, batch, D_loss.item(), G_loss.item()))

        # Save model checkpoints
        if (ep) % 10 == 0:
            save_checkpoint(ep)

    save_checkpoint(ep)

