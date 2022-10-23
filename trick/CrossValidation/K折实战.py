import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, SubsetRandomSampler, ConcatDataset
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms as transforms

warnings.filterwarnings('ignore')


class ConvNet(nn.Module):
    def __init__(self, h1=96):
        # We optimize dropout rate in a convolutional neural network.
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.drop1 = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(32 * 7 * 7, h1)
        self.drop2 = nn.Dropout2d(p=0.1)

        self.fc2 = nn.Linear(h1, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))

        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = self.drop1(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.drop2(x)

        x = self.fc2(x)

        return x


def train_epoch(model, device, dataloader, loss_fn, optimizer):

    train_loss, train_correct = 0.0, 0

    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()
    return train_loss, train_correct


def valid_epoch(model, device, dataloader, loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output, labels)
        valid_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        val_correct += (predictions == labels).sum().item()

    return valid_loss, val_correct


if __name__ == "__main__":

    train_dataset = torchvision.datasets.MNIST('classifier_data', train=True, download=True)
    test_dataset = torchvision.datasets.MNIST('classifier_data', train=False, download=True)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    train_dataset.transform = transform
    test_dataset.transform = transform

    m = len(train_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    criterion = nn.CrossEntropyLoss()

    dataset = ConcatDataset([train_dataset, test_dataset])

    num_epochs = 10
    batch_size = 128
    k = 10
    splits = KFold(n_splits=k, shuffle=True, random_state=42)
    foldperf = {}

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

        model = ConvNet()
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.002)

        history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

        for epoch in range(num_epochs):
            train_loss, train_correct = train_epoch(model, device, train_loader, criterion, optimizer)
            test_loss, test_correct = valid_epoch(model, device, test_loader, criterion)

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100

            print(
                "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} "
                "AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(
                    epoch + 1,
                    num_epochs,
                    train_loss,
                    test_loss,
                    train_acc,
                    test_acc))
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

        foldperf['fold{}'.format(fold + 1)] = history

    torch.save(model, 'k_cross_CNN.pt')

    # 评估K折交叉验证的效果
    testl_f, tl_f, testa_f, ta_f = [], [], [], []
    k = 10
    for f in range(1, k + 1):
        tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
        testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))
        ta_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
        testa_f.append(np.mean(foldperf['fold{}'.format(f)]['test_acc']))

    print('Performance of {} fold cross validation'.format(k))
    print("Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test "
          "Acc: {:.2f}".format(np.mean(tl_f), np.mean(testl_f), np.mean(ta_f), np.mean(testa_f)))
