import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import utils

CUDA = True     # Change to False for CPU training
OUT_PATH = 'output'
LOG_FILE = os.path.join(OUT_PATH, 'log.txt')
BATCH_SIZE = 16        # Adjust this value according to your GPU memory
IMAGE_CHANNEL = 1
Z_DIM = 100
G_HIDDEN = 64
X_DIM = 512
D_HIDDEN = 64
EPOCH_NUM = 500
REAL_LABEL = 1
FAKE_LABEL = 0
lr = 1.5e-4
seed = 1            # Change to None to get different results at each run


################################################################################
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPOINT = 'checkpoints'
LOAD_MODEL = True
lastEpoch = 342
CHECKPOINT_GEN = 'genCheckpoint_' + str(lastEpoch) + '.pth.tar'
CHECKPOINT_DISC = 'discrimCheckpoint_' + str(lastEpoch) + '.pth.tar'
################################################################################


utils.clear_folder(OUT_PATH)
print("Logging to {}\n".format(LOG_FILE))
sys.stdout = utils.StdOut(LOG_FILE)
CUDA = CUDA and torch.cuda.is_available()
print("PyTorch version: {}".format(torch.__version__))
if CUDA:
    print("CUDA version: {}\n".format(torch.version.cuda))

if seed is None:
    seed = np.random.randint(1, 10000)
print("Random Seed: ", seed)
np.random.seed(seed)
torch.manual_seed(seed)
if CUDA:
    torch.cuda.manual_seed(seed)
cudnn.benchmark = True      # May train faster but cost more memory

dataset = dset.ImageFolder( root="dataset/",transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=IMAGE_CHANNEL),
    transforms.Resize(X_DIM),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=4)

device = torch.device("cuda:0" if CUDA else "cpu")


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    

def weights_init(m):
    # custom weights initialization
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            #1
            nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 64, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 64),
            nn.ReLU(True),
            #4
            nn.ConvTranspose2d(G_HIDDEN * 64, G_HIDDEN * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 32),
            nn.ReLU(True),
            #8
            nn.ConvTranspose2d(G_HIDDEN * 32, G_HIDDEN * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 16),
            nn.ReLU(True),
            #16
            nn.ConvTranspose2d(G_HIDDEN * 16, G_HIDDEN * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 8),
            nn.ReLU(True),
            #32
            nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.ReLU(True),
            #64
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(True),
            #128
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU(True),
            #256
            nn.ConvTranspose2d(G_HIDDEN, IMAGE_CHANNEL, 4, 2, 1, bias=False),
            nn.Tanh()
            #512
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            #512
            nn.Conv2d(1, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            #256
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            #128
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            #64
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            #32
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            #16
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            #8
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            #4
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

netG = Generator().to(device)
netG.apply(weights_init)
print(netG)

netD = Discriminator().to(device)
netD.apply(weights_init)
print(netD)

criterion = nn.BCELoss()

viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))


if LOAD_MODEL:
    load_checkpoint(
        os.path.join(CPOINT, CHECKPOINT_GEN), netG, optimizerG, lr,
    )
    load_checkpoint(
        os.path.join(CPOINT, CHECKPOINT_DISC), netD, optimizerD, lr,
    )

for epoch in range( lastEpoch + 1, EPOCH_NUM + lastEpoch):
    for i, data in enumerate(dataloader):
        x_real = data[0].to(device)
        real_label = torch.full((x_real.size(0),), REAL_LABEL, device=device)
        fake_label = torch.full((x_real.size(0),), FAKE_LABEL, device=device)

        # Update loss_D_real
        netD.zero_grad()
        y_real = netD(x_real)
        real_label = real_label.float()
        loss_D_real = criterion(y_real, real_label)

        # Update loss_D_fake
        z_noise = torch.randn(x_real.size(0), Z_DIM, 1, 1, device=device)
        x_fake = netG(z_noise)
        y_fake = netD(x_fake.detach())
        fake_label = fake_label.float()
        loss_D_fake = criterion(y_fake, fake_label)

        # Calculate average loss and update discriminator
        loss_D_ave = (loss_D_real + loss_D_fake)/2
        loss_D_ave.backward()
        optimizerD.step()

        # Update G with fake data
        netG.zero_grad()
        y_fake_r = netD(x_fake)
        real_label = real_label.float()
        loss_G = criterion(y_fake_r, real_label)
        loss_G.backward()
        optimizerG.step()

        if i % 100 == 0:
            print('Epoch {} [{}/{}] loss_D_real: {:.4f} loss_D_fake: {:.4f} loss_G: {:.4f}'.format(
                epoch, i, len(dataloader),
                loss_D_real.mean().item(),
                loss_D_fake.mean().item(),
                loss_G.mean().item()
            ))
    vutils.save_image(x_real, os.path.join(OUT_PATH, 'real_samples.png'), normalize=True)
    with torch.no_grad():
        viz_noise = torch.randn(1, Z_DIM, 1, 1, device=device)
        viz_sample = netG(viz_noise)
        vutils.save_image(viz_sample, os.path.join(OUT_PATH, 'fake_samples_{}.png'.format(epoch)), normalize=True)
        
    ###########################################################################
    
    checkpointG = {
        "state_dict": netG.state_dict(),
        "optimizer": optimizerG.state_dict(),
    }
    torch.save(checkpointG, os.path.join(OUT_PATH, 'genCheckpoint_{}.pth.tar'.format(epoch)))
    
    checkpointD = {
        "state_dict": netD.state_dict(),
        "optimizer": optimizerD.state_dict(),
    }
    torch.save(checkpointD, os.path.join(OUT_PATH, 'discrimCheckpoint_{}.pth.tar'.format(epoch)))
