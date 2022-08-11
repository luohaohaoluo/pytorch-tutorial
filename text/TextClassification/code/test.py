import torch
import io
import torchtext
import time

from torch.optim import optimizer

from preprocess import MyTextDataset
from models import TextClassificationModel
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, vocab
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset


train_file_path = "../dataset/train.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ag_news_label = {1: "World",
                 2: "Sports",
                 3: "Business",
                 4: "Sci/Tec"}


def yield_tokens(file_path):
    with io.open(file_path, encoding='utf-8') as f:
        for line in f:
            yield tokenizer(line)


def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text)).to(device)
        output = model(text, torch.tensor([0]).to(device))
        return output.argmax(1).item() + 1


if __name__ == "__main__":
    ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
        enduring the season’s worst weather conditions on Sunday at The \
        Open on his way to a closing 75 at Royal Portrush, which \
        considering the wind and the rain was a respectable showing. \
        Thursday’s first round at the WGC-FedEx St. Jude Invitational \
        was another story. With temperatures in the mid-80s and hardly any \
        wind, the Spaniard was 13 strokes better in a flawless round. \
        Thanks to his best putting performance on the PGA Tour, Rahm \
        finished with an 8-under 62 for a three-stroke lead, which \
        was even more impressive considering he’d never played the \
        front nine at TPC Southwind."
    ex_text_str2 = "Ma Yun is a boss of the alibaba!!"

    train_iter = MyTextDataset(train_file_path, train=True)
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(train_file_path), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    text_pipeline = lambda x: vocab(tokenizer(x))

    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(vocab)
    em_size = 64
    model = TextClassificationModel(vocab_size, em_size, num_class)
    model.load_state_dict(torch.load("./models.pth"))
    model = model.to(device)

    print("This is a %s news" % ag_news_label[predict(ex_text_str2, text_pipeline)])
