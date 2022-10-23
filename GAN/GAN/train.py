import argparse
import os
import numpy as np
import math
import torchvision

import torch.cuda
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.utils import save_image
from models import *

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda = True if torch.cuda.is_available() else False

    os.makedirs("images", exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
    opt = parser.parse_args()

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    # Initialize generator and discriminator
    generator = Generator(opt.latent_dim, torch.tensor(img_shape)).to(device)
    discriminator = Discriminator(opt.latent_dim, torch.tensor(img_shape)).to(device)

    # Configure data loader
    os.makedirs("data/mnist", exist_ok=True)

    train_data = MNIST(root="data/mnist", train=True,
                       transform=transforms.Compose([transforms.Resize(opt.img_size),
                                                     transforms.ToTensor(), transforms.
                                                    Normalize([0.5], [0.5])])
                       )
    dataloader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = torch.ones((imgs.size(0), 1), requires_grad=False).to(device)
            fake = torch.zeros((imgs.size(0), 1), requires_grad=False).to(device)

            # Configure input
            real_imgs = imgs.type(Tensor).to(device)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.rand((imgs.shape[0], opt.latent_dim)).to(device)

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i % 200 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

    torch.save(generator.state_dict(), "models.pth")
