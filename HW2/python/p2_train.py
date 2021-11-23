import random
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from p2_model import Generator, Discriminator
from p2_dataset import DATA

batch_size = 64
epochs = 70

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

manualSeed = 1004
random.seed(manualSeed)
torch.manual_seed(manualSeed)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    trainset = DATA(root=os.path.join('data/digits/mnistm/'), transform=transform)
    traindataloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, pin_memory=True)

    image_criterion = nn.BCELoss()
    class_criterion = nn.CrossEntropyLoss()

    G = Generator().to(device)
    D = Discriminator().to(device)

    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for ep in range(epochs):
        for batch, (image, label) in enumerate(traindataloader, 1):
            Timage, Tlabel = image.to(device), label.to(device)

            noise = torch.randn(batch_size, 100, device=device)
            Flabel = torch.randint(0, 10, (batch_size,), device=device)
            fake = G(noise, Flabel)

            ################
            #   train G    #
            ################

            gd, gc = D(fake)
            Ttarget = torch.ones(batch_size, 1, device=device)
            gd_loss = image_criterion(gd, Ttarget)
            gc_loss = class_criterion(gc, Flabel)

            G_loss = (gd_loss * 0.5) + gc_loss
            g_optimizer.zero_grad()
            G_loss.backward()
            g_optimizer.step()

            ################
            #   train D    #
            ################
            noise = torch.randn(batch_size, 100, device=device)
            Flabel = torch.randint(0, 10, (batch_size,), device=device)
            fake = G(noise, Flabel)

            rd, rc = D(Timage)
            Ttarget = torch.ones(Timage.size(0), 1, device=device)
            rd_loss = image_criterion(rd, Ttarget)
            rc_loss = class_criterion(rc, Tlabel)

            fd, fc = D(fake.detach())
            Ftarget = torch.zeros(fake.size(0), 1, device=device)
            fd_loss = image_criterion(fd, Ftarget)
            fc_loss = class_criterion(fc, Flabel)

            D_loss = (rd_loss*0.5) + rc_loss + (fd_loss*0.5)
            d_optimizer.zero_grad()
            D_loss.backward(retain_graph=True)
            d_optimizer.step()


            if batch % 500 == 0 or batch == len(traindataloader):
                print('Epoch {} Iteration {}: discriminator_loss {:.3f} generator_loss {:.3f}'.format(ep, batch,D_loss.item(),G_loss.item()))

        if ep > 35:
            torch.save(G.state_dict(), 'checkpoint/{}-G.pth'.format(ep))
            print('Model saved!')