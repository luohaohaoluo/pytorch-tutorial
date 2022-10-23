import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        # opt.latent_dim
        super(Generator, self).__init__()

        self.img_shape = img_shape
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(torch.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.reshape(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Discriminator, self).__init__()

        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.reshape(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


if __name__ == "__main__":
    G = Generator(100, torch.tensor((2, 3, 28, 228)))
    D = Discriminator(100, torch.tensor((2, 3, 28, 228)))
    print(G)
    print(D)
    # for name, i in G.named_modules():
    #     print(name)
