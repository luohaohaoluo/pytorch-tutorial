import os
import numpy as np
import torch
import transforms as T
import torchvision
import utils

from models import fasterrcnn_fpn_resnet50
from preprocess import VOCDataset
from engine import train_one_epoch, evaluate
from PIL import Image


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 21
    # use our dataset and defined transformations
    dataset = VOCDataset('../dataset', get_transform(train=True), train_set=True)
    dataset_test = VOCDataset('../dataset', get_transform(train=False), train_set=False)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = fasterrcnn_fpn_resnet50(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset

        # evaluate(model, data_loader_test, device=device)
    torch.save(model.state_dict(), "./model.pth")

    print("That's it!")


if __name__ == '__main__':
    main()
