import io
import torchtext
import pandas as pd
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer


def extract_text(path, tok, train=True):
    """
    extracting the IBDM data info , and return the (texts,labels) belong to numpy
    :param path: the path of dataset contained the train.csv and test.csv
    :param tok:
    :param train: if True ,extract the train.csv; otherwise extract the text.csv
    :return: list(texts), labels
    """
    texts, labels = [], []
    if train:
        path = list(pathlib.Path(path).glob('train.csv'))
    else:
        path = list(pathlib.Path(path).glob('test.csv'))

    df = pd.read_csv(str(path[0]), encoding='utf-8')
    for item in range(len(df.index)):
        texts.append(tok(df.iloc[item, 0]))
        if df.iloc[item, 1] == 'neg':
            labels.append(0)
        else:
            labels.append(1)

    return texts, torch.LongTensor(labels)


class TextDataset(Dataset):

    def __init__(self, vectors, texts_list, labels_list, max_length):
        super(TextDataset, self).__init__()

        self.max_length = max_length
        self.vectors = vectors
        self.texts = texts_list
        self.labels = labels_list

    def __getitem__(self, item):
        temp = []
        label = self.labels[item]
        text = self.texts[item]
        if self.max_length < len(text):
            text = text[:self.max_length]
        else:
            distance = self.max_length - len(text)
            text[len(text):] = ['unk'] * distance

        for i in text:
            temp.append(self.vectors[i].numpy())
        temp = np.array(temp)
        temp = torch.tensor(temp)
        return temp, label

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    file_path = "../dataset"
    vec = torchtext.vocab.GloVe(name="840B", dim=300)
    tokenizer = get_tokenizer('basic_english')

    texts, labels = extract_text(file_path, tokenizer, train=True)
    # print(vec[text[0][0]])
    datasets = TextDataset(vec, texts, labels, max_length=40)
    train_loader = torch.utils.data.DataLoader(datasets, batch_size=32)

    print(train_loader.dataset[0])
    # text = extract_text(file_path, tokenizer, train=False)
    # dataset = torch.utils.data.TensorDataset(text, label)
    # print(dataset[0])








