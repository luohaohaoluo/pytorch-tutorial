from torch import nn

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1d = nn.Conv1d(embed_dim, 128, kernel_size=3, stride=2)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        x = self.embedding(x)
        # print("embedding:", x.shape)

        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        # print("conv1d:", x.shape)

        x = x.permute(0, 2, 1)
        x, (_h, _c) = self.lstm(x)
        # print("lstm:", x.shape)

        x = self.fc(x[:, -1, :])
        # print("fc:", x.shape)
        return x


if __name__ == '__main__':
    model = TextClassificationModel(100, 100, 4)
    print(model)

