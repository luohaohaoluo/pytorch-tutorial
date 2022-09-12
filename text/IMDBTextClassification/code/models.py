import torchtext
import torch
import torch.nn.functional as F
from torchtext.vocab import Vectors


class TextCNN(torch.nn.Module):
    def __init__(self, emb_size, num_classes):
        super(TextCNN, self).__init__()
        self.emb_size = emb_size
        self.num_classes = num_classes

        # # non-static
        # self.embedding = torch.nn.Embedding(len(vectors), emb_size)

        # use-static
        self.Conv2d1 = torch.nn.ModuleList([torch.nn.Conv2d(1, 100, kernel_size=(i, emb_size)) for i in range(3, 6)])
        # self.Conv2d1 = torch.nn.Conv2d(1, 100, kernel_size=(3, emb_size))
        # # use-non-static
        # self.Conv2d2 = torch.nn.ModuleList([torch.nn.Conv2d(1, 100, kernel_size=(i, emb_size)) for i in range(3, 6)])

        self.fc = torch.nn.Linear(17100, num_classes)

    def forward(self, x):
        # x --> [batch, sentence_length, vectors]
        temp = []
        for i, layer in enumerate(self.Conv2d1):
            temp.append(F.relu(layer(x)))

        x1 = torch.cat((temp[0], temp[1]), dim=2)
        x1 = torch.cat((x1, temp[2]), dim=2)

        x1 = torch.flatten(x1, start_dim=1, end_dim=-1)
        x1 = F.dropout(x1, 0.5)
        x1 = self.fc(x1)
        x1 = F.softmax(x1, 1)
        return x1


if __name__ == '__main__':
    # vec = torchtext.vocab.GloVe(name="840B", dim=300)
    # vec = Vectors('glove.840B.300d.txt')
    # print(vec['unk'])
    model = TextCNN(300, 2)
    print(model)

