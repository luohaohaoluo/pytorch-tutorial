import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from modules import EnDecoder

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    """
    1. 数据集的预处理
    """
    dataset_path = r"../../ImageClassification/dataset"
    train_dataset = FashionMNIST(dataset_path, train=True, download=True,
                                 transform=torchvision.transforms.ToTensor())

    test_dataset = FashionMNIST(dataset_path, train=False, download=True,
                                transform=torchvision.transforms.ToTensor())

    # 将图像数据转化为向量数据
    x_train = train_dataset.data.type(torch.float) / 255.0
    x_train = x_train.reshape(x_train.shape[0], -1)
    y_train = train_dataset.targets

    x_test = test_dataset.data.type(torch.float) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1)
    y_test = test_dataset.targets

    train_dataloader = DataLoader(x_train, batch_size=64)
    print("训练数据集：", x_train.shape)
    print("测试数据集: ", x_test.shape)

    """
    2. 训练集部分可视化
    """
    plt.figure(figsize=(6, 6))
    # 可视化一个batch的图像内容
    for step, x in enumerate(train_dataloader):
        if step > 0:
            break

        im = make_grid(x.reshape(-1, 1, 28, 28))
        im = im.data.numpy().transpose(1, 2, 0)
        plt.imshow(im)
        plt.axis('off')
        plt.show()

    """
    3.创建模型并训练
    """
    module = EnDecoder()

    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()  #可以BCELoss

    train_loss_all = []

    module.train()
    for epoch in range(10):
        tran_loss = 0
        train_num = 0

        for step, X in enumerate(train_dataloader):
            _, output = module(X)
            loss = loss_fn(output, X,)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tran_loss += loss.item() * X.size(0)
            train_num += X.size(0)
        train_loss_all.append(tran_loss / train_num)

    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(range(1, 11), train_loss_all)
    plt.savefig("./result.jpg")

    torch.save(module.state_dict(), "./model.pth")












