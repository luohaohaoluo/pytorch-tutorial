import argparse
import copy

import torchvision
import time
import torch
import torch.nn.functional as F

from tqdm import tqdm
from models import ViT
from torch import optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir='./log')

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.003, help="adam: learning rate")
    parser.add_argument("--download_path", type=str, default='./dataset', help="download the dataset into dir")
    opt = parser.parse_args()

    train_dataset = MNIST(opt.download_path, train=True, download=True,
                          transform=torchvision.transforms.ToTensor())

    test_dataset = MNIST(opt.download_path, train=False, download=True,
                         transform=torchvision.transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)

    EPOCHS = opt.epochs

    '''
    patch大小为 7x7（对于 28x28 图像，这意味着每个图像 4 x 4 = 16 个patch）、10 个可能的目标类别（0 到 9）和 1 个颜色通道（因为图像是灰度）。
    在网络参数方面，使用了 64 个单元的维度，6 个 Transformer 块的深度，8 个 Transformer 头，MLP 使用 128 维度。'''
    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=6, heads=8, mlp_dim=128).to(device)

    dummy_input = torch.rand(1, 1, 28, 28).to(device)
    writer.add_graph(model, input_to_model=dummy_input)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    '''
    train（loss and accuracy）
    '''

    for epoch in range(EPOCHS):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), colour='#259DA1', leave=True, unit='batch')
        '''
        train_loader
        '''
        for ind, (X, Y) in loop:
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()

            output = model(X)
            loss = loss_fn(output, Y)

            loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch [{epoch}/{EPOCHS}]')
            loop.set_postfix(loss=loss.item())
            if ind == len(train_loader) - 1:
                writer.add_scalar("train_loss", loss.item(), epoch)
        '''
        test_loader
        '''
        all_accnum, all_len = [], []
        for ind, (X, Y) in enumerate(test_loader):
            X, Y = X.to(device), Y.to(device)
            all_len.append(len(Y))

            output = model(X)
            pre = sum(torch.argmax(output, dim=1) == Y)
            all_accnum.append(pre)

        acc = sum(all_accnum) / sum(all_len)
        print(f"\n{epoch}: the test accuracy is {acc:.2%}")
        writer.add_scalar("test_acc", acc, epoch)

    writer.close()
    torch.save(model.state_dict(), "./models.pt")

