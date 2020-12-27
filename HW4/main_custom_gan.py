import argparse
import os
import math
import time

import cv2
import glob
import imageio
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensor
from tqdm import tqdm
from os.path import join as pjoin

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from torch import nn, optim
import torch.nn.functional as F
import torch

from scipy.stats import truncnorm

from data_loader import prepare_loader
from custom_gan import Generator, Discriminator, PixelwiseNorm, MinibatchStdDev

SEED = 42
torch.manual_seed(SEED)

# For fast training
torch.backends.cudnn.benchmark = True

def train(opt):
    os.makedirs("result/images_train", exist_ok=True)
    os.makedirs("result/models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loss function
    criterion = nn.BCELoss()

    # Initialize generator and discriminator
    netG = Generator(opt.latent_dim, 32, opt.channels).to(device)
    netD = Discriminator(opt.channels, 48).to(device)

    # Config dataloader
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

    train_loader = torch.utils.data.DataLoader(DogDataset('data/images_crop'),
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=opt.n_cpu)

    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    lr_schedulerG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerG,
                                                                        T_0=opt.n_epochs//200, eta_min=0.00005)
    lr_schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerD,
                                                                        T_0=opt.n_epochs//200, eta_min=0.00005)
    # fixed_noise = torch.randn(25, opt.latent_dim, 1, 1, device=device)

    real_label = 0.7
    fake_label = 0.0

    # ----------
    #  Training
    # ----------

    step = 0
    for epoch in tqdm(range(opt.n_epochs)):
        total_d_loss, total_g_loss, total_D_x, total_D_G_z1, total_D_G_z2 = 0, 0, 0, 0, 0
        for ii, (real_images) in enumerate(train_loader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            labels = torch.full((batch_size, 1), real_label, device=device) +  np.random.uniform(-0.1, 0.1)

            output = netD(real_images)
            errD_real = criterion(output, labels)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, opt.latent_dim, 1, 1, device=device)
            fake = netG(noise)
            labels.fill_(fake_label) + np.random.uniform(0, 0.2)
            output = netD(fake.detach())
            errD_fake = criterion(output, labels)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            labels.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, labels)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            total_g_loss += errG.item()
            total_d_loss += errD.item()
            total_D_x += D_x
            total_D_G_z1 += D_G_z1
            total_D_G_z2 += D_G_z2
            
            if step % opt.sample_interval == 0:
                save_image(fake.data[:25],
                            "result/images_train/%d.png" % step,
                            nrow=5,
                            normalize=True)
            step += 1
            lr_schedulerG.step(epoch)
            lr_schedulerD.step(epoch)

        print("[Epoch %d/%d] [D loss: %.5f] [G loss: %.5f] [D(x): %.5f] [D(G(z)): %.5f / %.5f]" %
                (epoch, opt.n_epochs, total_d_loss / len(train_loader), total_g_loss / len(train_loader), total_D_x / len(train_loader), total_D_G_z1 / len(train_loader), total_D_G_z2 / len(train_loader)))

    # Save final model
    os.makedirs("result/models/final", exist_ok=True)
    torch.save(netD.state_dict(),
                "result/models/final/discriminator.pt")
    torch.save(netG.state_dict(),
                "result/models/final/generator.pt")


def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values


def inference(opt):
    os.makedirs("result/images_inference", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize generator and discriminator
    netG = Generator(opt.latent_dim, 32, opt.channels).to(device)

    # Load trained models
    netG.load_state_dict(torch.load(opt.model_path + '/generator.pt'))

    # Inference
    for i in tqdm(range(opt.inference_num)):
        z = truncated_normal((1, opt.latent_dim, 1, 1), threshold=1)
        gen_z = torch.from_numpy(z).float().to(device)
        gen_images = netG(gen_z)
        images = gen_images.to("cpu").clone().detach()
        images = images.numpy().transpose(0, 2, 3, 1)

        save_image((gen_images+1.0)/2.0, "result/images_inference/%d.png" % i)


def process_data(opt):
    """ Processes data according to bbox in annotations. """

    os.makedirs("data/images_crop", exist_ok=True)
    os.makedirs("data/images_ref", exist_ok=True)
    cuda = True if torch.cuda.is_available() else False

    # Config dataloader
    root_images = pjoin(opt.data_path, 'images/') #'data/images/'
    root_annots = pjoin(opt.data_path, 'annotations/') #'data/annotations/'
    dataloader = prepare_loader(root_images, root_annots, opt.batch_size)

    # Save cropped images
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    image_count = 0
    for imgs in tqdm(dataloader):
        real_imgs = Variable(imgs.type(Tensor))
        for i in range(0, len(real_imgs)):
            save_image(real_imgs.data[i],
                       "data/images_crop/%d.png" % image_count,
                       normalize=True)

            # Save first K images as reference for calculating FID scores.
            if image_count < opt.inference_num:
                save_image(real_imgs.data[i],
                           "data/images_ref/%d.png" % image_count,
                           normalize=True)
            image_count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        choices=['train', 'inference', 'process_data'],
                        required=True,
                        help="operation mode")
    parser.add_argument("--model_path",
                        type=str,
                        default='result/models/final',
                        help="model path for inference")
    parser.add_argument("--data_path",
                        type=str,
                        default='data',
                        help="data path for data process and training")
    parser.add_argument("--n_epochs",
                        type=int,
                        default=200,
                        help="number of epochs of training")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="size of the batches")
    parser.add_argument("--lr",
                        type=float,
                        default=2e-4,
                        help="adam: learning rate")
    parser.add_argument("--b1",
                        type=float,
                        default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2",
                        type=float,
                        default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim",
                        type=int,
                        default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--img_size",
                        type=int,
                        default=64,
                        help="size of each image dimension")
    parser.add_argument("--channels",
                        type=int,
                        default=3,
                        help="number of image channels")
    parser.add_argument("--sample_interval",
                        type=int,
                        default=400,
                        help="interval between image sampling")
    parser.add_argument("--inference_num",
                        type=int,
                        default=10000,
                        help="number of generated images for inference")
    parser.add_argument("--n_cpu", 
                        type=int, 
                        default=4, 
                        help="number of cpu threads to use during batch generation")

    opt = parser.parse_args()

    global img_shape
    img_shape = (opt.channels, opt.img_size, opt.img_size)
    
    print(opt)

    if opt.mode == 'train':
        train(opt)
    elif opt.mode == 'inference':
        inference(opt)
    else:
        process_data(opt)
