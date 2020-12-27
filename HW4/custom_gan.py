import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm_notebook as tqdm
from time import time
from PIL import Image
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.image as mpimg
import torchvision
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import xml.etree.ElementTree as ET
import random
from torch.nn.utils import spectral_norm
from scipy.stats import truncnorm
import torch as th

import argparse
import math
import time

import cv2
import glob
import imageio
import albumentations as A
from albumentations.pytorch import ToTensor
from tqdm import tqdm
from os.path import join as pjoin

# batch_size = 32

# def seed_everything(seed=42):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
# seed_everything()

class DogDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        super().__init__()

        #self.images = None
        self.images = []

        # Read all png files
        for im_path in tqdm(glob.glob(pjoin(path, "*.png"))):
            im = imageio.imread(im_path)
            im = np.array(im, dtype='float')
            self.images.append(im)

        self.images = np.array(self.images)
        self.transform = A.Compose(
            [A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            #  A.augmentations.transforms.ColorJitter(brightness=0.2, contrast=0.9, saturation=0.3, hue=0.01, p=0.2),
                ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = self.transform(image=img)['image']

        return img

# train_loader = torch.utils.data.DataLoader(DogDataset('data/images_crop'),
#                                             batch_size=batch_size,
#                                             shuffle=True,
#                                             num_workers=8)

class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y

class MinibatchStdDev(th.nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape
        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)
        # [1 x C x H x W]  Calc standard deviation over batch
        y = th.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size,1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = th.cat([x, y], 1)
        # return the computed values:
        return y


class Generator(nn.Module):
    def __init__(self, nz, nfeats, nchannels):
        super(Generator, self).__init__()

        # input is Z, going into a convolution
        self.conv1 = spectral_norm(nn.ConvTranspose2d(nz, nfeats * 8, 4, 1, 0, bias=False))
        # state size. (nfeats*8) x 4 x 4
        
        self.conv2 = spectral_norm(nn.ConvTranspose2d(nfeats * 8, nfeats * 8, 4, 2, 1, bias=False))
        # state size. (nfeats*8) x 8 x 8
        
        self.conv3 = spectral_norm(nn.ConvTranspose2d(nfeats * 8, nfeats * 4, 4, 2, 1, bias=False))
        # state size. (nfeats*4) x 16 x 16
        
        self.conv4 = spectral_norm(nn.ConvTranspose2d(nfeats * 4, nfeats * 2, 4, 2, 1, bias=False))
        # state size. (nfeats * 2) x 32 x 32
        
        self.conv5 = spectral_norm(nn.ConvTranspose2d(nfeats * 2, nfeats, 4, 2, 1, bias=False))
        # state size. (nfeats) x 64 x 64
        
        self.conv6 = spectral_norm(nn.ConvTranspose2d(nfeats, nchannels, 3, 1, 1, bias=False))
        # state size. (nchannels) x 64 x 64
        self.pixnorm = PixelwiseNorm()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.pixnorm(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.pixnorm(x)
        x = F.leaky_relu(self.conv4(x))
        x = self.pixnorm(x)
        x = F.leaky_relu(self.conv5(x))
        x = self.pixnorm(x)
        x = torch.tanh(self.conv6(x))
        
        return x


class Discriminator(nn.Module):
    def __init__(self, nchannels, nfeats):
        super(Discriminator, self).__init__()

        # input is (nchannels) x 64 x 64
        self.conv1 = nn.Conv2d(nchannels, nfeats, 4, 2, 1, bias=False)
        # state size. (nfeats) x 32 x 32
        
        self.conv2 = spectral_norm(nn.Conv2d(nfeats, nfeats * 2, 4, 2, 1, bias=False))
        self.bn2 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats*2) x 16 x 16
        
        self.conv3 = spectral_norm(nn.Conv2d(nfeats * 2, nfeats * 4, 4, 2, 1, bias=False))
        self.bn3 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 8 x 8
       
        self.conv4 = spectral_norm(nn.Conv2d(nfeats * 4, nfeats * 8, 4, 2, 1, bias=False))
        self.bn4 = nn.MaxPool2d(2)
        # state size. (nfeats*8) x 4 x 4
        self.batch_discriminator = MinibatchStdDev()
        self.pixnorm = PixelwiseNorm()
        self.conv5 = spectral_norm(nn.Conv2d(nfeats * 8 +1, 1, 2, 1, 0, bias=False))
        # state size. 1 x 1 x 1
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.batch_discriminator(x)
        x = torch.sigmoid(self.conv5(x))
        return x.view(-1, 1)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# lr = 0.0002
# lr_d = 0.0002
# beta1 = 0.5
# epochs = 200
# netG = Generator(100, 32, 3).to(device)
# netD = Discriminator(3, 48).to(device)

# criterion = nn.BCELoss()

# optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=lr_d, betas=(beta1, 0.999))
# lr_schedulerG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerG,
#                                                                      T_0=epochs//200, eta_min=0.00005)
# lr_schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerD,
#                                                                      T_0=epochs//200, eta_min=0.00005)

# nz = 100
# fixed_noise = torch.randn(25, nz, 1, 1, device=device)

# real_label = 0.7
# fake_label = 0.0
# batch_size = train_loader.batch_size

# os.makedirs("result/images_train", exist_ok=True)
# os.makedirs("result/models", exist_ok=True)

# ### training here


# step = 0
# for epoch in range(epochs):
#     for ii, (real_images) in enumerate(train_loader):
#         ############################
#         # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
#         ###########################
#         # train with real
#         netD.zero_grad()
#         real_images = real_images.to(device)
#         batch_size = real_images.size(0)
#         labels = torch.full((batch_size, 1), real_label, device=device) +  np.random.uniform(-0.1, 0.1)

#         output = netD(real_images)
#         errD_real = criterion(output, labels)
#         errD_real.backward()
#         D_x = output.mean().item()

#         # train with fake
#         noise = torch.randn(batch_size, nz, 1, 1, device=device)
#         fake = netG(noise)
#         labels.fill_(fake_label) + np.random.uniform(0, 0.2)
#         output = netD(fake.detach())
#         errD_fake = criterion(output, labels)
#         errD_fake.backward()
#         D_G_z1 = output.mean().item()
#         errD = errD_real + errD_fake
#         optimizerD.step()

#         ############################
#         # (2) Update G network: maximize log(D(G(z)))
#         ###########################
#         netG.zero_grad()
#         labels.fill_(real_label)  # fake labels are real for generator cost
#         output = netD(fake)
#         errG = criterion(output, labels)
#         errG.backward()
#         D_G_z2 = output.mean().item()
#         optimizerG.step()
        
#         if step % 500 == 0:
#             print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
#                   % (epoch + 1, epochs, ii, len(train_loader),
#                      errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
#             valid_image = netG(fixed_noise)
#             save_image(valid_image.data[:25],
#                         "result/images_train/%d.png" % step,
#                         nrow=5,
#                         normalize=True)
#         step += 1
#         lr_schedulerG.step(epoch)
#         lr_schedulerD.step(epoch)


# # Save final model
# os.makedirs("result/models/final", exist_ok=True)
# torch.save(netG.state_dict(),
#             "result/models/final/discriminator.pt")
# torch.save(netD.state_dict(),
#             "result/models/final/generator.pt")

# def truncated_normal(size, threshold=1):
#     values = truncnorm.rvs(-threshold, threshold, size=size)
#     return values

# if not os.path.exists('../output_images'):
#     os.mkdir('../output_images')
# im_batch_size = 100
# n_images=10000
# for i_batch in range(0, n_images, im_batch_size):
#     z = truncated_normal((im_batch_size, 100, 1, 1), threshold=1)
#     gen_z = torch.from_numpy(z).float().to(device)    
#     #gen_z = torch.randn(im_batch_size, 100, 1, 1, device=device)
#     gen_images = netG(gen_z)
#     images = gen_images.to("cpu").clone().detach()
#     images = images.numpy().transpose(0, 2, 3, 1)
#     for i_image in range(gen_images.size(0)):
#         save_image((gen_images[i_image, :, :, :] +1.0)/2.0, os.path.join('../output_images', f'image_{i_batch+i_image:05d}.png'))


# import shutil
# shutil.make_archive('images', 'zip', '../output_images')