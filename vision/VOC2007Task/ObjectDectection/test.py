import os
import numpy as np
import torch

import transforms as T
import torchvision
import utils

from models import fasterrcnn_fpn_resnet50
from preprocess import VOCDataset
from PIL import Image, ImageDraw, ImageFont

labels_map = {
        "aeroplane": 1,
        "bicycle": 2,
        "bird": 3,
        "boat": 4,
        "bottle": 5,
        "bus": 6,
        "car": 7,
        "cat": 8,
        "chair": 9,
        "cow": 10,
        "diningtable": 11,
        "dog": 12,
        "horse": 13,
        "motorbike": 14,
        "person": 15,
        "pottedplant": 16,
        "sheep": 17,
        "sofa": 18,
        "train": 19,
        "tvmonitor": 20
    }


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():

    labels_temp = {str(value): key for key, value in labels_map.items()}

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 21
    # use our dataset and defined transformations
    dataset_test = VOCDataset('../dataset', get_transform(train=False), train_set=False)

    # get the model using our helper function
    model = fasterrcnn_fpn_resnet50(num_classes)
    model.load_state_dict(torch.load("./model.pth"))

    # torch.save(model, "model_all.pth")
    # move model to the right device
    model.to(device)

    img, target = dataset_test[20]
    real_box = target['boxes'].numpy()

    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

        boxes = prediction[0]["boxes"].cpu().numpy()
        scores = prediction[0]["scores"].cpu().numpy()
        labels = prediction[0]["labels"].cpu().numpy()
        step = 0
        for score in scores:
            if score > 0.7:
                step += 1

        boxes = boxes[0:step, :]
        labels = labels[0:step]
        scores = scores[0:step]
        real_label = []
        for i in labels:
            real_label.append(labels_temp.get(str(i)))

        im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        draw = ImageDraw.Draw(im)

        for boxe, label, score in zip(boxes, real_label, scores):
            draw.rectangle(boxe, outline=(0, 255, 0), width=2)
            draw.rectangle((boxe[0], boxe[1]-20, boxe[0]+70, boxe[1]), outline=(0, 255, 0), fill=(0, 255, 0), width=2)
            draw.text((boxe[0]+3, boxe[1]-20), label, fill=(255, 255, 255))
            draw.text((boxe[0]+3, boxe[1]-10), str(score), fill=(255, 255, 255))
        # for boxe in real_box:
        #     draw.rectangle(boxe, outline=(0, 0, 256), width=2)
        im.show()


if __name__ == "__main__":
    main()
