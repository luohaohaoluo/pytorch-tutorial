import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
from modules import EnDecoder

dataset_path = r"../../ImageClassification/dataset"

test_dataset = FashionMNIST(dataset_path, train=False, download=True,
                            transform=torchvision.transforms.ToTensor())

x_test = test_dataset.data.type(torch.float) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1)
y_test = test_dataset.targets

module = EnDecoder()
module.load_state_dict(torch.load("./model.pth"))
module.eval()

_, test_decoder = module(x_test[0:100, :])

plt.figure(figsize=(6, 6))
for ii in range(test_decoder.shape[0]):
    plt.subplot(10, 10, ii+1)
    im = x_test[ii, :]
    im = im.data.numpy().reshape(28, 28, 1)
    plt.imshow(im, cmap='gray')
    plt.axis('off')
plt.show()


plt.figure(figsize=(6, 6))
for ii in range(test_decoder.shape[0]):
    plt.subplot(10, 10, ii+1)
    im = test_decoder[ii, :]
    im = im.detach().numpy().reshape(28, 28, 1)
    plt.imshow(im, cmap='gray')
    plt.axis('off')
plt.show()





