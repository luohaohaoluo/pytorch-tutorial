import torch.nn
from tqdm import tqdm


from preprocess import *
from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    file_path = "../dataset"
    EPOCHS = 10
    batch_size = 32
    vec = torchtext.vocab.GloVe(name="840B", dim=300)
    tokenizer = get_tokenizer('basic_english')

    texts, labels = extract_text(file_path, tokenizer, train=True)
    datasets = TextDataset(vec, texts, labels, max_length=60)
    train_loader = torch.utils.data.DataLoader(datasets, batch_size=batch_size)

    model = TextCNN(300, 2).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), colour='#259DA1', leave=True, unit='batch')
        for ind, (X, Y) in loop:
            X = X.unsqueeze(dim=1).to(device)
            Y = Y.to(device)

            output = model(X)
            loss = loss_fn(output, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch [{epoch}/{EPOCHS}]')
            loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "./model.pth")


