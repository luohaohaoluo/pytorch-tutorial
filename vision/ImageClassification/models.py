import torch
import torch.nn as nn
from torchvision.models import vgg13

from torch.nn import Conv2d, MaxPool2d, Flatten, ReLU, Linear,Tanh


class ImageClassificationModule(nn.Module):
    """
    - the input = 24 × 24
    """
    def __init__(self, kernel_size=3, input_channels=3, output_channels=10):
        super(ImageClassificationModule, self).__init__()
        self.kernel_size = kernel_size
        self.input_channles = input_channels
        self.output_channles = output_channels

        self.feature = nn.Sequential(
            Conv2d(in_channels=self.input_channles, out_channels=16, kernel_size=self.kernel_size, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),    # --> (14,14)
            Conv2d(in_channels=16, out_channels=32, kernel_size=self.kernel_size, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),  # --> (7,7)
            Conv2d(in_channels=32, out_channels=32, kernel_size=self.kernel_size, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),  # --> (3,3)
        )
        # self.flatten = Flatten()
        self.classifier = nn.Sequential(
            Linear(in_features=32*3*3, out_features=32),
            Tanh(),
            Linear(in_features=32, out_features=64),
            Tanh(),
            Linear(in_features=64, out_features=self.output_channles)
        )

    def forward(self, input):
        output = self.feature(input)
        # output = self.flatten(output)
        output = torch.flatten(output, start_dim=1)
        output = self.classifier(output)
        return output


if __name__ == "__main__":
    model = ImageClassificationModule()
    for i in model.children():
        for j in i.children():
            print(j)


