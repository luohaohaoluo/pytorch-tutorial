from torch.utils.data import Dataset
import os
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor, Compose

"""
- the Dataset class can generate the data with returning the (img, label)
- root_dir : usually named dataset include train and val or test
- labels_dir : Under the train or val directory, 
                it should have some directory which is about classification
- train: if True, it means that you search the train dataset; Otherwise, searching the val dataset 
"""


class MyDataset(Dataset):
    def __init__(self, root_dir, labels_dir, *, train=True):
        super(MyDataset, self).__init__()
        self.root_dir = root_dir
        self.imgLib = []
        self.labels_dir = labels_dir
        self.transforms = Compose([ToTensor()])

        if(train == True):
            self.flag = 'train'
            self.data_dir = os.path.join(self.root_dir, self.flag)
            #  /hymenoptera_data/train

            self.img_path = Path(self.data_dir)
            self.imgLib.extend(list(self.img_path.glob('*/*.jp*g')))

        elif(train == False):
            self.flag = 'val'
            self.data_dir = os.path.join(self.root_dir, self.flag)
            #  /hymenoptera_data/train

            self.img_path = Path(self.data_dir)

            self.imgLib.extend(list(self.img_path.glob('*/*.jp*g')))

    def __getitem__(self, item):
        img_path = str(self.imgLib[item])
        print(img_path)

        """
        Modify the labels_dir[num] to generate the label's number
        """
        label = 0 if self.labels_dir[0] in img_path.split('\\') else 1
        if label == 1:
            label = 1 if self.labels_dir[1] in img_path.split('\\') else 2

        img = Image.open(img_path)
        # img.show()
        img = self.transforms(img)

        return img, label

    def __len__(self):
        image_count = len(list(self.img_path.glob('*/*.jp*g')))
        print(f"The {self.flag} dataset has {image_count} pictures")
        return image_count


if __name__ == "__main__":

    root_dir = "../hymenoptera_data"
    labels_dir = ['ants', 'bees']

    dataset = MyDataset(root_dir=root_dir, labels_dir=labels_dir, train=True)
    print(dataset[126])
    print(len(dataset))