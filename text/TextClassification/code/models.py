from torch import nn

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.num_layers = 2
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
#         self.lstm = nn.LSTM(embed_dim, 64, self.num_layers, bidirectional=True)
        self.gru = nn.GRU(embed_dim, 64, self.num_layers, bidirectional=True)
        self.fc = nn.Linear(128, num_class)
        self.init_weights()
#         # lstm parameters
#         self.h0 = torch.randn(2*self.num_layers, barch_size, 128)
#         self.c0 = torch.randn(2*self.num_layers, barch_size, 128)
#         # gru parameters
#         self.h1 = torch.randn(2*self.num_layers, barch_size, 256)
#         self.c1 = torch.randn(2*self.num_layers, barch_size, 256)

    def init_weights(self):
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        x = self.embedding(text, offsets)
#         x = self.lstm(x, (self.h0,self.c0))
#         x = self.gru(x, (self.h1,self.c1))
#         x,(_h,_c) = self.lstm(x)
        x, _h = self.gru(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = TextClassificationModel(100, 100, 4)
    print(model)
