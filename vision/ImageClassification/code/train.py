"""
Image Classification
"""
import torch
from torchvision.datasets import FashionMNIST
import torchvision
import matplotlib.pyplot as plt
from models import *
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter



batch_size = 32
epochs = 10
learning_rate = 0.001


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter("./logs")

labels = {'0': "T-shirt/top",  # T恤
          '1': 'Trouser',  # 裤子
          '2': 'Pullover',  # 套衫
          '3': 'Dress',  # 裙子
          '4': 'Coat',  # 外套
          '5': 'Sandal',  # 凉鞋
          '6': 'Shirt',  # 衬衫
          '7': 'Sneaker',  # 运动鞋
          '8': 'Bag',  # 包
          '9': 'Ankle boot'  # 短靴
          }


def draw_picture(dataset, label, index=0):
    """
    require : numpy
    :param dataset: the picture belong to the dataset
    :param label: labels of the class
    :param index: the location of picture
    :return: None , but show the picture and print some info of the pic
    """
    plt.imshow(torch.squeeze(dataset[index][0]), cmap='gray')
    plt.show()
    print(dataset[index][0].shape)
    print(label[str(dataset[index][1])])


def train_loop(dataloader, model, loss_fn, optimizer, step):
    size = len(dataloader.dataset)
    # num_batches = len(dataloader)
    model.train()
    train_loss, train_correct = 0, 0

    # train_acc_all, train_loss_all = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        y.data.type(torch.float32)
        X, y = X.to(device), y.to(device)

        length = len(X)
        # Compute prediction and loss
        pred = model(X)
        train_loss = loss_fn(pred, y)

        train_correct = (pred.argmax(1) == y).sum().item()
        # train_acc_all += train_correct
        # train_loss_all += train_loss

        # Backpropagation
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_correct /= length
        if batch % 200 == 0:
            loss, current = train_loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} accuracy: {train_correct:>7f} [{current:>5d}/{size:>5d}]")
            writer.add_scalar("train_loss", train_loss.item(), step)
            writer.add_scalar("train_accuracy", train_correct, step)
            step += 1

    # train_acc_all /= size
    # train_loss_all /= num_batches
    # print(f"Train Finish: \n Accuracy: {(100 * train_acc_all):>0.1f}%, Avg loss: {train_loss_all:>8f} \n")
    return step

def test_loop(dataloader, model, loss_fn, step):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    test_loss, test_correct = 0, 0

    test_loss_all, test_accuracy_all = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            length = len(X)
            pred = model(X)
            test_loss = loss_fn(pred, y).item()
            test_correct = (pred.argmax(1) == y).sum().item()

            test_loss_all += test_loss
            test_accuracy_all += test_correct

            test_correct /= length
            if batch % 200 == 0:
                writer.add_scalar("test_accuracy", test_correct, step)
                writer.add_scalar("test_loss", test_loss, step)
                step += 1

    test_loss_all /= num_batches
    test_accuracy_all /= size
    print(f"Test Finish: \n Accuracy: {(100 * test_accuracy_all):>0.1f}%, Avg loss: {test_loss_all:>8f} \n")
    return step


if __name__ == "__main__":
    train_data = FashionMNIST(root="../dataset", train=True,
                              transform=torchvision.transforms.ToTensor(),
                              download=True)
    test_data = FashionMNIST(root="../dataset", train=False,
                             transform=torchvision.transforms.ToTensor(),
                             download=False)

    draw_picture(test_data, labels, 12)

    print(f"Using {device} device")

    model = ImageClassificationModule(kernel_size=3, input_channels=1, output_channels=10)
    model = model.to(device)

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    train_step = 0
    test_step = 0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_step = train_loop(train_dataloader, model, loss_fn, optimizer, train_step)
        test_step = test_loop(test_dataloader, model, loss_fn, test_step)
    print("Done!")

    writer.close()

    torch.save(model.state_dict(), "./model.pth")
    print("Saved PyTorch Model State to model.pth")
