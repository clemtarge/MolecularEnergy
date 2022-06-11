import argparse
from statistics import mean

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import numpy as np

import pickle

import pandas as pd

from net import Conv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
 
    def __init__(self,file_name, start_split = 0, end_split = -1, y_min = None, y_max = None):

        with open(file_name, "rb") as file:
            df = pickle.load(file)
    
        self.x_pos = df.iloc[start_split:end_split,0].values
        self.x_num = df.iloc[start_split:end_split,2].values
        y_df = df.iloc[start_split:end_split,3].values
        if y_min is None:
            self.y_min = np.min(y_df)
        else: 
            self.y_min = y_min
        if y_max is None:
            self.y_max = np.max(y_df)
        else:
            self.y_max = y_max
        self.y = (y_df - self.y_min)/(self.y_max - self.y_min)
 
    def __len__(self):
        return len(self.y)
   
    def __getitem__(self,idx):
        return torch.tensor([self.x_pos[idx]],dtype=torch.float32), \
                torch.tensor([self.x_num[idx]],dtype=torch.float32), \
                torch.tensor([self.y[idx]],dtype=torch.float32)

def train(net, optimizer, loader, epochs=10):#, writer, epochs=10):
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x_pos, x_num, y in t:
            x_pos, x_num, y = x_pos.to(device), x_num.to(device), y.to(device)
            outputs = net(x_pos, x_num)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss):.16f}')
        #writer.add_scalar('training loss', mean(running_loss), epoch)

def test(model, dataloader):
    cur_MSE = 0
    with torch.no_grad():
        for x_pos, x_num, y in dataloader:
            print(x_pos.shape)
            x_pos, x_num, y = x_pos.to(device), x_num.to(device), y.to(device)
            y_hat = model(x_pos, x_num)
            cur_MSE += nn.MSELoss()(y_hat, y)

    return cur_MSE/len(dataloader)

def evaluate(model, file_name, y_min, y_max):
    with open(file_name, "rb") as file:
        df = pickle.load(file)

    x_pos = df.iloc[:,0].values
    x_num = df.iloc[:,2].values
    y_hat = model(torch.tensor(x_pos), torch.tensor(x_num))*(y_max - y_min) + y_min

    results = pd.DataFrame({"id": list(df.index), "energy": y_hat.detach().numpy()})

    return results

if __name__=='__main__':

    # TODO: writer = SummaryWriter(f'runs/MNIST')
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='kaggle', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='training epochs')

    args = parser.parse_args()
    exp_name = args.exp_name
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    # datasets
    traindata = MyDataset('train_preprocess.pkl', end_split=5336)
    testdata = MyDataset('train_preprocess.pkl', start_split=5336, \
                        y_min = traindata.y_min, y_max = traindata.y_max)

    # dataloaders
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True)

    net = Conv()

    # setting net on device(GPU if available, else CPU)
    net = net.to(device)
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train(net, optimizer, trainloader, epochs)#, writer, epochs)
    test_acc = test(net, testloader)
    print(f'Test accuracy:{test_acc}')
    torch.save(net.state_dict(), "mnist_net.pth")


    truetestdata = evaluate(net, 'truetest_preprocess.pkl', \
                        y_min = traindata.y_min, y_max = traindata.y_max)

    truetestdata.to_csv('truetest_pred.csv', index=False)

    #y_hat = model(x_pos, x_num)

    # # add embeddings to tensorboard
    # perm = torch.randperm(len(trainset.data))
    # images, labels = trainset.data[perm][:256], trainset.targets[perm][:256]
    # images = images.unsqueeze(1).float().to(device)
    # with torch.no_grad():
    #     embeddings = net.get_features(images)
    #     writer.add_embedding(embeddings,
    #                          metadata=labels,
    #                          label_img=images, global_step=1)

    # # save networks computational graph in tensorboard
    # writer.add_graph(net, images)
    # # save a dataset sample in tensorboard
    # img_grid = torchvision.utils.make_grid(images[:64])
    # writer.add_image('mnist_images', img_grid)

