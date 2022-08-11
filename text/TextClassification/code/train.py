import torch
import io
import torchtext
import time

from torch.optim import optimizer

from preprocess import MyTextDataset
from models import TextClassificationModel
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, vocab
from torchtext.data.functional import to_map_style_dataset


train_file_path = "../dataset/train.csv"
test_file_path = "../dataset/test.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}

# Hyper parameters
epoch = 10  # epoch
barch_size = 64  # batch size for training
LR = 5  # learning rate


def yield_tokens(file_path):
    with io.open(file_path, encoding='utf-8') as f:
        for line in f:
            yield tokenizer(line)


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


def train(dataloader, loss_fn):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = loss_fn(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader, loss_fn):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = loss_fn(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


if __name__ == "__main__":

    '''
    1. Access to the raw dataset iterators and and Generate data batch and iterator
    '''
    # train_iter, test_iter = AG_NEWS()
    train_iter = MyTextDataset(train_file_path, train=True)
    test_iter = MyTextDataset(test_file_path, train=False)
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(split_train_, batch_size=barch_size,
                                  shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=barch_size,
                                  shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=barch_size,
                                 shuffle=True, collate_fn=collate_batch)

    '''
    2. Prepare data processing pipelines  
    '''
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(train_file_path), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    # print(vocab["<unk>"])
    # print(vocab(['here', 'is', 'the', 'an', 'example']))
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1
    # print(torch.tensor(text_pipeline('here is the an example')).size(0))
    # print(label_pipeline('10'))

    '''
    3. Define the model
    '''
    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(vocab)
    em_size = 64
    model = TextClassificationModel(vocab_size, em_size, num_class).to(device)

    '''
    4. Train the model
    '''
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
    total_accu = None

    for epoch in range(1, epoch + 1):
        epoch_start_time = time.time()
        train(train_dataloader, criterion)
        accu_val = evaluate(valid_dataloader,criterion)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               accu_val))
        print('-' * 59)
        print('Checking the results of test dataset.')
        accu_test = evaluate(test_dataloader, criterion)
        print('test accuracy {:8.3f}'.format(accu_test))

    '''
    5.save the model
    '''
    torch.save(model.state_dict(), "./models.pth")


