import torch.nn as nn
from torchvision.models import vgg19_bn, vgg19,resnet50
from torchsummary import summary


class FCN_Vgg_8s(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        model_vgg19 = vgg19(pretrained=True)

        # model_vgg19 = vgg19_bn(pretrained=True)
        self.backbone = model_vgg19.features

        self.relu = nn.ReLU(inplace=True)
        self.de_conv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1,
                                           dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)

        self.de_conv2 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1, 1)
        self.bn2 = nn.BatchNorm2d(256)

        self.de_conv3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.de_conv4 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, 1)
        self.bn4 = nn.BatchNorm2d(64)

        self.de_conv5 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

        self.layers = {"4": "maxpool_1", "9": "maxpool_2", "18": "maxpool_3",
                       "27": "maxpool_4", "36": "maxpool_5"}

    def forward(self, x):
        output = {}
        for name, layer in self.backbone._modules.items():
            x = layer(x)
            if name in self.layers:
                output[self.layers[name]] = x
        x5 = output["maxpool_5"]
        x4 = output["maxpool_4"]
        # print(x4.shape)
        x3 = output["maxpool_3"]

        score = self.relu(self.de_conv1(x5))
        # 如果使用带BN的, x4.shape=[4, 512, 40, 60], score.shape=[4, 512, 80, 120]
        score = self.bn1(score + x4)
        score = self.relu(self.de_conv2(score))
        score = self.bn2(score + x3)
        score = self.bn3(self.relu(self.de_conv3(score)))
        score = self.bn4(self.relu(self.de_conv4(score)))
        score = self.bn5(self.relu(self.de_conv5(score)))
        score = self.classifier(score)
        return score


if __name__ == "__main__":

    model = FCN_Vgg_8s(21)
    # print(model.backbone)
    summary(model, input_size=(3, 320, 480))
    # model = resnet50(pretrained=True)
    # print(model)

