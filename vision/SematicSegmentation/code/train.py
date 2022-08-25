import copy
from models import FCN_Vgg_8s
from preprocess import *
from torch.nn.functional import log_softmax

Batch_size = 8
LR = 3e-5
EPOCH = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_name = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "dining table",
              "dog", "horse", "motorbike", "person", "potted plant",
              "sheep", "sofa", "train", "tv/monitor"]

label_color = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
               [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
               [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
               [0, 192, 0], [128, 192, 0], [0, 64, 128]
               ]


def train_model(model, criterion, optimizer, train_dataloader, val_dataloader, num_epochs):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print('-' * 50)
        train_loss = 0.0
        train_num = 0
        val_loss = 0.0
        val_num = 0

        model.train()
        for batch_id, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = x.float().to(device)
            y = y.long().to(device)
            # print(x.shape)
            out = model(x)
            out = log_softmax(out, dim=1)
            # print(out.shape)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            train_num += len(y)
        # 计算一个epoch在训练集上的loss
        train_loss_all.append(train_loss / train_num)
        print(f'epoch{epoch} Train loss: {train_loss_all[-1]:.4f}')

        # 计算一个epoch在验证集上的loss
        model.eval()
        for batch_id, (x, y) in enumerate(val_dataloader):
            optimizer.zero_grad()
            x = x.float().to(device)
            y = y.long().to(device)
            out = model(x)
            out = log_softmax(out, dim=1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            val_loss += loss.item() * len(y)
            val_num += len(y)
        # 计算一个epoch在训练集上的loss
        val_loss_all.append(val_loss / val_num)
        print(f'epoch{epoch} Val loss: {val_loss_all[-1]:.4f}')

        # 保存最好的参数
        if val_loss_all[-1] < best_loss:
            best_loss = val_loss_all[-1]
            best_model_weights = copy.deepcopy(model.state_dict())

    torch.save(best_model_weights, './model.pth')


if __name__ == '__main__':
    H, W = 320, 480
    train_dataset = MyDataset('../dataset/VOC2012/ImageSets/Segmentation/train.txt',
                              H, W, data_transform, label_color)
    val_dataset = MyDataset('../dataset/VOC2012/ImageSets/Segmentation/val.txt',
                            H, W, data_transform, label_color)
    train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=True)

    model = FCN_Vgg_8s(21).to(device)
    loss_fn = torch.nn.NLLLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    train_model(model, loss_fn, opt, train_loader, val_loader, EPOCH)


