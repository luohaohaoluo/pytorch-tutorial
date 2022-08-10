import torch.nn as nn


class EnDecoder(nn.Module):
    def __init__(self):
        super(EnDecoder, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 16),
            nn.Tanh()
        )

        self.Decoder = nn.Sequential(
            nn.Linear(16, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return encoder, decoder


if __name__ == "__main__":
    module = EnDecoder()
    print(module)


