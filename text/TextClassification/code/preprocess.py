from abc import ABC

import pandas as pd
import pathlib
import numpy as np
import torch


class MyTextDataset(torch.utils.data.IterableDataset):
    def __init__(self, csv_path, train=True):
        super(MyTextDataset, self).__init__()
        name = 'train' if train else 'test'
        self.data_file = pd.read_csv(csv_path, encoding='utf-8', sep=',')
        self.labels = list(self.data_file["label"])
        self.texts = list(self.data_file["text"])
        print(f"the {name} dataset are {len(self.data_file.index)}'s numbers")

    def __iter__(self):
        return iter(zip(self.labels, self.texts))

    def __next__(self):
        return next(iter(zip(self.labels, self.texts)))

    def __len__(self):
        return len(self.data_file.index)


def convert_dataset(raw_path: str, new_path: str, train: bool = True):
    """
    convert the ag_news train.csv or test.csv(contain 3 columns:Class Index,Title,Description) \
        generate a new train.csv or test.csv(contain 2 columns:Class Index,Description)
    notice!! the Description contains the Title

    args:
    - raw_path: input a dataset path
    - new_path: save new train.csv or test.csv into here
    """

    name = 'train' if train else 'test'
    raw_path = pathlib.Path(raw_path)
    raw_path = list(raw_path.glob(f"*{name}.csv"))[0]
    raw_path = str(raw_path)

    raw_file = pd.read_csv(raw_path)
    # print(len(raw_file.index))
    rows = raw_file.shape[0]
    columns = raw_file.shape[1] - 1

    new_path = new_path + f"/{name}.csv"
    # print(new_path)
    new_file = pd.DataFrame(np.zeros((rows, columns)), index=np.arange(rows), columns=["label", "text"])
    new_file["label"] = raw_file["Class Index"]
    new_file["text"] = raw_file["Title"]
    new_file["text"] += raw_file["Description"]

    # x = raw_file["Title"] + raw_file["Description"]
    # print(x)
    new_file.to_csv(new_path)


if __name__ == "__main__":
    file_path = "../dataset"
    convert_dataset(file_path, file_path, train=True)
    convert_dataset(file_path, file_path, train=False)
    train_iter = MyTextDataset("../dataset/train.csv")
    # print(len(train_iter))
    print(next(train_iter))
