import torch
from torchvision.datasets import FashionMNIST
import torchvision
import matplotlib.pyplot as plt
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print(f"the picture is {label[str(dataset[index][1])]}")

def test_fn(model, dataset, device, label, *, index):
    input_tensor = torch.unsqueeze(dataset[index][0], dim=0)
    # print(input_tensor.shape)
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)
    pre = output.argmax(1).item()

    print(f"raw label is :{dataset[index][1]}\npredict lable is {pre}")
    draw_picture(dataset, label, index=index)

if __name__ == "__main__":
    test_data = FashionMNIST(root="../dataset", train=False,
                             transform=torchvision.transforms.ToTensor(),
                             download=False)

    model = ImageClassificationModule(kernel_size=3, input_channels=1, output_channels=10)
    model.load_state_dict(torch.load("./model.pth"))
    model = model.to(device)

    print(model.classifier[0])
    print(model)
    indexs = 306
    test_fn(model, test_data, device, labels, index=indexs)
    draw_picture(test_data, labels, index=indexs)
