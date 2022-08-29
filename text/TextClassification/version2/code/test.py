import torch
import io

from models import TextClassificationModel
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, vocab


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
        text = torch.tensor(text_pipeline(text)).unsqueeze(dim=0).to(device)
        output = model(text)
        return output.argmax(1).item() + 1


if __name__ == "__main__":

    ex_text_str1 = "Bingtian Su is a superstar in sports!"
    ex_text_str2 = "HuaWei is a famous company!!"

    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens("../dataset/train.csv"), specials=["<unk>"], max_tokens=1000, min_freq=3)
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))

    num_class = 4
    vocab_size = len(vocab)
    em_size = 64
    model = TextClassificationModel(vocab_size, em_size, num_class).to(device)
    model.load_state_dict(torch.load("models_new.pth"))

    print(f"\'{ex_text_str1}\'  is a %s news" % ag_news_label[predict(ex_text_str1, text_pipeline)])
    print(f"\'{ex_text_str2}\' is a %s news" % ag_news_label[predict(ex_text_str2, text_pipeline)])
