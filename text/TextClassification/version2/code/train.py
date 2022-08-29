import copy
import time

import torch
import io

import torchtext

from preprocess import TextDataset
from models import TextClassificationModel
from torch.utils.data import random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

train_file_path = "../dataset/train.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}

# Hyper parameters
epochs = 10  # epoch
barch_size = 64  # batch size for training
LR = 1e-3  # learning rate


def yield_tokens(file_path):
    with io.open(file_path, encoding='utf-8') as f:
        for line in f:
            yield tokenizer(line)


if __name__ == "__main__":

    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens("../dataset/train.csv"), specials=["<unk>"], max_tokens=1000, min_freq=3)
    vocab.set_default_index(vocab["<unk>"])

    train_dataset = TextDataset("../dataset/train.csv", 30, vocab, tokenizer)
    num = int(len(train_dataset) * 0.95)
    train_dataset, val_dataset = random_split(train_dataset, [num, len(train_dataset) - num])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=barch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=barch_size)

    num_class = 4
    vocab_size = len(vocab)
    # print(vocab_size)
    em_size = 64
    model = TextClassificationModel(vocab_size, em_size, num_class).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.001)

    for epoch in range(epochs):
        best_loss = 10
        best_weights = copy.deepcopy(model.state_dict())
        print("-------train-------")
        model.train()
        # train_acc_all, train_loss_all = [], []
        for idx, (label, text) in enumerate(train_loader):
            label = label.to(device)
            text = text.to(device)
            count = len(label)

            output = model(text)
            loss = loss_fn(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            train_acc = (output.argmax(1) == label).sum().item()
            train_acc /= count

            if idx % 500 == 0:
                print(f"|epoch:{epoch} | batches: {idx}/{len(train_loader)} | loss: {train_loss:.4f}: | accuracy: {train_acc:.4%}")

        print("-------validation-------")
        model.eval()
        val_loss_all = []
        val_loss, val_acc, val_num = 0, 0, 0
        with torch.no_grad():
            for idx, (label, text) in enumerate(val_loader):
                label = label.to(device)
                text = text.to(device)
                count = len(label)

                output = model(text)
                loss = loss_fn(output, label)

                val_loss += loss.item()
                val_acc += (output.argmax(1) == label).sum().item()
                val_num += count

        val_loss_all.append(val_loss/val_num)
        print("-" * 50)
        print(f"epoch:{epoch}: | loss: {val_loss_all[-1]:.4f}: | accuracy: {val_acc / val_num:.4%}")
        print("-" * 50)
        if best_loss > val_loss_all[-1]:
            best_loss = val_loss_all[-1]
            best_weights = copy.deepcopy(model.state_dict())

    torch.save(best_weights, "./models_new.pth")

