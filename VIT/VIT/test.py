import torch
import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from models import ViT
from torchvision.transforms import ToTensor, Resize, Compose


def test_single_pic(mod, raw_img, img_transform):
    """

    :param mod: model
    :param raw_img:  img (Image.open --> PIL)
    :param img_transform: (transform)
    :return: result of single pic
    """
    img_tensor = img_transform(raw_img)
    img_tensor = img_tensor.unsqueeze(dim=0)
    output = mod(img_tensor)
    output = torch.nn.functional.softmax(output, dim=1)
    pro = torch.max(output)
    pre = torch.argmax(output, dim=1)
    return pre, pro


if __name__ == "__main__":
    pic_dir = os.listdir("./testpic")  #

    all_pre = []
    transform = Compose([Resize((28, 28)), ToTensor()])

    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=6, heads=8, mlp_dim=128)
    model.load_state_dict(torch.load("./models.pt", map_location='cpu'))

    plt.figure(figsize=(5, 3))
    for ind, img_name in enumerate(pic_dir):
        img = Image.open(f"./testpic/{img_name}").convert('L')  #
        pre, pro = test_single_pic(model, img, transform)
        print(pro)
        plt.subplot(1, 3, ind+1)

        img = np.array(img)
        plt.imshow(img, cmap='gray')
        plt.text(0, -2, f"pre:{int(pre.detach().numpy())},pro:{float(pro.detach().numpy()):.1%}")
        plt.axis('off')

    # plt.savefig('./result.png')
    plt.show()







