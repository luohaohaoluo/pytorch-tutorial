from preprocess import *
from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    file_path = "../dataset"
    EPOCHS = 10
    vec = torchtext.vocab.GloVe(name="840B", dim=300)
    tokenizer = get_tokenizer('basic_english')

    texts, labels = extract_text(file_path, tokenizer, train=False)
    datasets = TextDataset(vec, texts, labels, max_length=60)

    model = TextCNN(300, 2).to(device)
    model.load_state_dict(torch.load("./model.pth"))
    count = 0
    for (X, Y) in datasets:
        X = X.unsqueeze(0).unsqueeze(0).to(device)
        Y = Y.to(device)
        output = model(X)
        pre = output.argmax(1)
        if Y == pre:
            count += 1

    print(count / len(datasets))
    # 0.749





    # temp = []
    # test_list = tokenizer(test_str2)
    # test_list = test_list[:60]
    # for i in test_list:
    #     temp.append(vec[i].numpy())
    # temp = np.array(temp)
    # temp = torch.tensor(temp).to(device)
    # temp = temp.unsqueeze(0).unsqueeze(0)
    #
    # model = TextCNN(300, 2).to(device)
    # model.load_state_dict(torch.load("./model.pth"))
    #
    # output = model(temp)
    #
    # pre = output.argmax(1)
    # print("the news belong to {}".format('negative' if pre == 0 else 'positive'))





