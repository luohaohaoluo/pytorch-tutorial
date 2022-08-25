import numpy as np
import matplotlib.pyplot as plt
from models import FCN_Vgg_8s
from preprocess import *
from torch.nn.functional import log_softmax

Batch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

label_color = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
               [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
               [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
               [0, 192, 0], [128, 192, 0], [0, 64, 128]
               ]

if __name__ == "__main__":
    H, W = 320, 480
    val_dataset = MyDataset('../dataset/VOC2012/ImageSets/Segmentation/val.txt',
                            H, W, data_transform, label_color)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=True)
    model = FCN_Vgg_8s(21).to(device)
    model.load_state_dict(torch.load("./model.pth"))
    model.eval()
    for batch_id, (x, y) in enumerate(val_loader):

        x = x.float().to(device)
        y = y.long().to(device)
        out = model(x)
        out = log_softmax(out, dim=1)
        pre_lab = torch.argmax(out, 1)
        # can't convert cuda:0 device type tensor to numpy, 要加tensor.cpu()
        x_numpy = x.cpu().data.numpy()
        x_numpy = x_numpy.transpose(0, 2, 3, 1)
        y_numpy = y.cpu().data.numpy()
        pre_numpy = pre_lab.cpu().data.numpy()
        plt.figure(figsize=(16, 9))
        for ii in range(4):
            plt.subplot(3, 4, ii+1)
            plt.imshow(inv_normalize_image(x_numpy[ii]))
            plt.axis('off')

            plt.subplot(3, 4, ii + 5)
            plt.imshow(label2image(y_numpy[ii], label_color))
            plt.axis('off')

            plt.subplot(3, 4, ii + 9)
            # temp = label2image(pre_numpy[ii], label_color)
            plt.imshow(label2image(pre_numpy[ii], label_color))
            plt.axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()
        break







