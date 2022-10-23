import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

"""
残差网络堆叠注意事项：
    - 从阶段的角度：第二阶段 即conv2_x 不需要下采样，之后的阶段都需要下采样！！
    - 从堆叠残差块的角度： 每个阶段堆叠的残差块中，只有第一个残差块需要下采样！！
"""


class Basicblock(nn.Module):
    """ ResNet 18 34 使用的残差块 """

    # 用于控制残差块中最后一个卷积层输出通道的倍率
    expansion = 1  # 通道膨胀率

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """

        :param in_channel:  输入通道
        :param out_channel: 输出通道
        :param stride:      控制第一个卷积层的步长，stride=2，说明主路需要下采样
        :param downsample:  控制shortcut是否需要下采样（与stride搭配使用）
        """

        super(Basicblock, self).__init__()

        # stride=1 (H,W)不变；否则(H,W)减半
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)

        # (H,W)不变
        # 引入通道膨胀率
        self.conv2 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identify = x
        if self.downsample is not None:
            identify = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identify
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ ResNet 50 101 152 使用的残差块 """

    # 用于控制残差块中最后一个卷积层输出通道的倍率
    expansion = 4  # 通道膨胀率

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """

        :param in_channel:  输入通道
        :param out_channel: 输出通道
        :param stride:      控制第一个卷积层的步长，stride=2，说明主路需要下采样
        :param downsample:  控制shortcut是否需要下采样（与stride搭配使用）
        """

        super(Bottleneck, self).__init__()

        # (H,W)不变
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)

        # stride=1 (H,W)不变；否则(H,W)减半
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)

        # (H,W)不变
        # 引入通道膨胀
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identify = x
        if self.downsample is not None:
            identify = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identify
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ 组合残差块，构成残差网络 """

    def __init__(self, block, block_num, num_classes):
        """

        :param block: 残差块种类: Basicblock or Bottleneck
        :param block_num: 一个list，表示每层堆叠多少残差块
        :param num_classes: 分类类别
        """
        super(ResNet, self).__init__()

        self.in_channel = 64  # 记录每个阶段，卷积的输入通道数

        # (3, 224, 224) --> (64, 112, 112)
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)

        # (64, 112, 112) --> (64, 56, 56)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # (64, 56, 56) --> (256, 56, 56)
        self.layer1 = self._make_layer(block=block, channel=64, block_num=block_num[0],
                                       stride=1)

        # (256, 56, 56) --> (512, 28, 28)
        self.layer2 = self._make_layer(block=block, channel=128, block_num=block_num[1],
                                       stride=2)

        # (512, 28, 28) --> (1024, 14, 14)
        self.layer3 = self._make_layer(block=block, channel=256, block_num=block_num[2],
                                       stride=2)

        # (1024, 14, 14) --> (2048, 7, 7)
        self.layer4 = self._make_layer(block=block, channel=512, block_num=block_num[3],
                                       stride=2)

        #  (2048, 7, 7) -->  (2048, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512 * block.expansion, out_features=num_classes)

        self._weight_init()

    def _make_layer(self, block, channel, block_num, stride=1):
        """

        :param block: 残差块种类
        :param channel: 不同阶段的堆叠模块中，第一个卷积的通道数
        :param block_num: 当前阶段的模块堆叠数量
        :param stride: 卷积步长，stride=2 表明需要下采样
        :return: 返回block_num个block
        """
        downsample = None

        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride,
                          padding=0, bias=False),
                nn.BatchNorm2d(num_features=channel * block.expansion)
            )

        layers = []

        # 每个阶段堆叠的残差块
        layers.append(block(self.in_channel, channel, stride=stride, downsample=downsample))

        # self.in_channel 会保存当前值，逐渐递增
        # 如conv2_x 输入通道是64， 那么conv2_x 输入通道会保存为64 * block.expansion
        self.in_channel = channel * block.expansion

        # 堆叠残差块除第一个外，其余的不需要下采样
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes=1000, pretrained=True):
    resnet18_url = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
    module = ResNet(block=Basicblock, block_num=[2, 2, 2, 2], num_classes=num_classes)
    if pretrained:
        module.load_state_dict(load_state_dict_from_url(resnet18_url))

    return module


def resnet50(num_classes=1000, pretrained=True):
    resnet50_url = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    module = ResNet(block=Bottleneck, block_num=[3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        module.load_state_dict(load_state_dict_from_url(resnet50_url))

    features = list([module.conv1, module.bn1, module.relu, module.maxpool, module.layer1, module.layer2, module.layer3, module.layer4,])
    classifier = list([module.avgpool])

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)

    return features, classifier


if __name__ == "__main__":
    # features, classifier = resnet50(pretrained=True)
    # print(features)
    model = resnet18(pretrained=True)
    print(model)
