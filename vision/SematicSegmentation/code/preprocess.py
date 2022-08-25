import torch
import pathlib
import numpy as np
import torchvision
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize

label_name = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
              "dog", "horse", "motorbike", "person", "potted plant",
              "sheep", "sofa", "train", "tv/monitor"]

label_color = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
               [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
               [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
               [0, 192, 0], [128, 192, 0], [0, 64, 128]
               ]


def image2label(image, colormap):
    """
    Converting the class[like(128, 0, 0)]-img to label(0~20)-img
    :param image:
    :param colormap:
    :return: 2D-numpy array
    """
    temp = np.zeros(256 ** 3)
    # 将每个像素类别存放在对应像素值索引处
    for i, cm in enumerate(colormap):
        temp[((cm[0] * 256 + cm[1]) * 256 + cm[2])] = i
    image = np.array(image, dtype='int64')
    # 生成像素类别的索引
    index = ((image[:, :, 0] * 256 + image[:, :, 1]) * 256 + image[:, :, 2])
    # 利用索引，提取像素对应类别，将类别2D张量取出
    return temp[index]


def data_random_crop(img, label, height, width):
    """
    Randomly Cropping the img and label, and keeping the pixel (img, label) is corresponding
    :param img: PIL-Image
    :param label: PIL-Image
    :param height: will saving high
    :param width: will saving width
    :return: PIL-img, PIL-label
    """
    # PIL的图片打开后是 (W, H)
    im_width, im_height = img.size
    # 生成图像随机点位置
    left = np.random.randint(0, im_width - width)
    top = np.random.randint(0, im_height - height)
    img = F.crop(img, top=top, left=left, height=height, width=width)
    label = F.crop(label, top=top, left=left, height=height, width=width)
    return img, label


def data_transform(img, label, height, width, colormap):
    """
    Transforming the PIL-Image to Tensor and converting the label-img to (high, width)
    :param img:  PIL-Image
    :param label: PIL-Image
    :param height: will saving high
    :param width: will saving width
    :param colormap: label-colormap
    :return:  img(tensor), label(tensor)
    """
    img, label = data_random_crop(img, label, height, width)
    img_transforms = Compose([
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    img = img_transforms(img)
    label = torch.from_numpy(image2label(label, colormap))
    return img, label


def read_image_path(root='../dataset/VOC2012/ImageSets/Segmentation/train.txt'):
    """ Saving path under root """
    image = np.loadtxt(root, dtype=str)
    n = len(image)
    data_path, label_path = [None] * n, [None] * n
    for i, name in enumerate(image):
        data_path[i] = f'../dataset/VOC2012/JPEGImages/{name}.jpg'
        label_path[i] = f'../dataset/VOC2012/SegmentationClass/{name}.png'
    return data_path, label_path


class MyDataset(Dataset):
    def __init__(self, data_root, height, width, img_transform, colormap):
        """
        :param data_root: data_path
        :param height: will saving high
        :param width: will saving width
        :param img_transform:
        :param colormap:
        """
        super(MyDataset, self).__init__()
        self.data_root = data_root
        self.height = height
        self.width = width
        self.img_transform = img_transform
        self.colormap = colormap
        data_path, label_path = read_image_path(root=data_root)
        self.data_path = self._filter(data_path)
        self.label_path = self._filter(label_path)

    def _filter(self, images):
        """ filter the picture if the size less than (H, W) """
        return [im for im in images if (Image.open(im).size[1] > self.height
                                        and Image.open(im).size[0] > self.width)]

    def __getitem__(self, item):
        img = self.data_path[item]
        label = self.label_path[item]
        img = Image.open(img)
        label = Image.open(label).convert("RGB")
        img, label = self.img_transform(img, label, self.height, self.width, self.colormap)
        return img, label

    def __len__(self):
        return len(self.data_path)


"""
----------------------验证随机裁剪是否出问题（原图和标签图是否对应上了）-------------------
"""


def inv_normalize_image(data):
    """ inverse the image normalizing """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    data = data.astype('float32') * std + mean
    # 限制data数据的 min=0, max=1
    return data.clip(0, 1)


def label2image(label, colormap):
    """ about label-img(0-20) and convert it to class-img (128, 0, 0)"""
    h, w = label.shape
    label = label.reshape(h*w, -1)
    image = np.zeros((h*w, 3), dtype='int32')
    for ii in range(len(colormap)):
        index = np.where(label == ii)
        image[index, :] = colormap[ii]
    return image.reshape(h, w, 3)


if __name__ == "__main__":
    H, W = 320, 480
    train_dataset = MyDataset('../dataset/VOC2012/ImageSets/Segmentation/train.txt',
                              H, W, data_transform, label_color)
    val_dataset = MyDataset('../dataset/VOC2012/ImageSets/Segmentation/val.txt',
                            H, W, data_transform, label_color)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for step, (b_x, b_y) in enumerate(train_loader):
        print("b_x.shape", b_x.shape)
        print("b_y,shape", b_y.shape)
        b_x_numpy = b_x.data.numpy()
        b_x_numpy = b_x_numpy.transpose(0, 2, 3, 1)
        b_y_numpy = b_y.data.numpy()
        plt.figure(figsize=(16, 6))
        for ii in range(4):
            plt.subplot(2, 4, ii+1)
            plt.imshow(inv_normalize_image(b_x_numpy[ii]))
            plt.axis('off')
            plt.subplot(2, 4, ii+5)
            plt.imshow(label2image(b_y_numpy[ii], label_color))
            plt.axis('off')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()
        break


