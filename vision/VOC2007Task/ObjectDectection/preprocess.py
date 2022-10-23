import os
import torch
import json
import transforms as T


from torch.utils.data import Dataset, DataLoader
from PIL import Image
from lxml import etree


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# 创建自己的dataset
class VOCDataset(Dataset):

    def __init__(self, voc_root, transforms, train_set=True):
        self.root = os.path.join(voc_root, 'VOC2007')
        self.img_root = os.path.join(self.root, 'JPEGImages')
        self.annotations_root = os.path.join(self.root, 'Annotations')

        if train_set:
            txt_list = os.path.join(self.root, 'ImageSets', 'Segmentation', 'train.txt')
        else:
            txt_list = os.path.join(self.root, 'ImageSets', 'Segmentation', 'val.txt')

        with open(txt_list) as read:
            # strip去掉换行符 得到所有标注（xml）文件的路径
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + '.xml') for line in read.readlines()]

        # 打开每一个类别所对应索引的json文件
        try:
            json_file = open('../dataset/pascal_voc_classes.json', 'r')
            # {'name': index}
            self.class_dict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # 读取当前idx下的xml文件
        xml_path = self.xml_list[idx]

        with open(xml_path) as read:
            xml_str = read.read()
        xml = etree.fromstring(xml_str)
        # parse_xml_to_dict 解析xml文件 -> dict
        # 若有多个目标 将object组成一个list[] 通过for循环取出每一个类别以及对应的坐标
        data = self.parse_xml_to_dict(xml)['annotation']
        # 将路径与图像名称拼接起来
        img_path = os.path.join(self.img_root, data['filename'])
        image = Image.open(img_path)
        if image.format != 'JPEG':
            raise ValueError('image not JPEG')

        boxes = []
        labels = [] # 存入的是类别所对应的索引值
        iscrowd = []
        # 可能含有多个目标
        for obj in data['object']:
            xmin = float(obj['bndbox']['xmin'])
            ymin = float(obj['bndbox']['ymin'])
            xmax = float(obj['bndbox']['xmax'])
            ymax = float(obj['bndbox']['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj['name']])
            iscrowd.append(int(obj['difficult']))  # 0 容易 1困难

        # 转换为tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)['annotation']
        data_height = int(data['size']['height'])
        data_width = int(data['size']['width'])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        # xml -> dict
        if len(xml) == 0:
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            # child.tag: filename -> child_result: {'filename': '2020_005183'} -> result{}
            # folder: VOC2012
            child_result = self.parse_xml_to_dict(child)
            # child.tag取出子目录的名称 判断是否为 object. eg: folfer filename...
            if child.tag != 'object':
                # key: folder value: VOC2012 ...
                result[child.tag] = child_result[child.tag]
            else:
                # object:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])

        return {xml.tag: result}


if __name__ == "__main__":
    data_path = '../dataset'
    dataset = VOCDataset(data_path, transforms=get_transform(train=True))
    print(dataset[2])
    # data_loader = DataLoader(
    #     dataset, batch_size=2, shuffle=True, num_workers=4,
    #     collate_fn=utils.collate_fn)
    # for batch_id, (img, target) in enumerate(data_loader):
    #     print(type(img[0]))
    #     break

