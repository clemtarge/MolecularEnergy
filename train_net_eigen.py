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

from net_eigen import Conv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
 
    def __init__(self,file_name, start_split = 0, end_split = -1, transform = None, is_test = False):

        with open(file_name, "rb") as file:
            df = pickle.load(file).sort_index()
    
        self.is_test = is_test
        self.x_pos = df.iloc[start_split:end_split,0].values
        self.x_atom_number = df.iloc[start_split:end_split,1].values
        self.x_n_atom = df.iloc[start_split:end_split,2].values

        if not self.is_test:
            self.y = df.iloc[start_split:end_split,3].values

        # min_list = []
        # max_list = []
        # for pos in self.x_pos:
        #     min_list.append(np.min(pos))
        #     max_list.append(np.max(pos))

        # if transform is None:
        #     self.transform = transforms.Normalize(np.min(min_list), np.max(max_list)-np.min(min_list))
        # else:
        #     self.transform = transform
 
    def __len__(self):
        return len(self.x_pos)
   
    def __getitem__(self,idx):
        # return_array = [self.transform(torch.tensor([self.x_pos[idx].reshape(23,23)],dtype=torch.float32))]
        return_array = [torch.tensor([self.x_pos[idx]],dtype=torch.float32),
                        torch.tensor([self.x_atom_number[idx]],dtype=torch.float32),
                        torch.tensor([self.x_n_atom[idx]],dtype=torch.float32).unsqueeze(0)]
        if not self.is_test:
            return_array.append(torch.tensor([self.y[idx]],dtype=torch.float32).unsqueeze(0))
            
        #print(return_array[0].shape)
        return return_array

def train(net, optimizer, loader, epochs=10):#, writer, epochs=10):
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x_pos, x_atom_number, x_n_atom, y in t:
            x_pos, x_atom_number, x_n_atom, y = x_pos.to(device), x_atom_number.to(device), x_n_atom.to(device), y.to(device)
            outputs = net(x_pos, x_atom_number, x_n_atom)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'epoch : {epoch:3d} | training loss: {mean(running_loss):4.16f}')
        #writer.add_scalar('training loss', mean(running_loss), epoch)

def test(model, dataloader):
    cur_MSE = 0
    with torch.no_grad():
        for x_pos, x_atom_number, x_n_atom, y in dataloader:
            x_pos, x_atom_number, x_n_atom, y = x_pos.to(device), x_atom_number.to(device), x_n_atom.to(device), y.to(device)
            y_hat = model(x_pos, x_atom_number, x_n_atom)
            cur_MSE += nn.MSELoss()(y_hat, y)

    return cur_MSE/len(dataloader)

def evaluate(model, dataloader, index_range = None):

    y_hat = []
    with torch.no_grad():
        for x_pos, x_atom_number, x_n_atom in dataloader:
            x_pos, x_atom_number, x_n_atom = x_pos.to(device), x_atom_number.to(device), x_n_atom.to(device)
            y_hat += model(x_pos, x_atom_number, x_n_atom).detach().tolist()

    if index_range is None:
        ids = range(len(y_hat))
    else:
        ids = range(index_range[0],index_range[1]+1)

    results = pd.DataFrame({"id": list(ids), "predicted": np.ravel(y_hat)[1:-3]})

    return results

if __name__=='__main__':

    # TODO: writer = SummaryWriter(f'runs/MNIST')
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='kaggle', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='training epochs')

    args = parser.parse_args()
    exp_name = args.exp_name
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    # datasets
    traindata = MyDataset('train_preprocess_modele_ensemble.pkl', end_split=5336)
    testdata = MyDataset('train_preprocess_modele_ensemble.pkl', start_split=5336)#, transform=traindata.transform)
    truetestdata = MyDataset('truetest_preprocess_modele_ensemble.pkl', is_test = True)#transform=traindata.transform)

    # dataloaders
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True)
    truetestloader = DataLoader(truetestdata, batch_size=batch_size, shuffle=False)

    net = Conv()

    # setting net on device(GPU if available, else CPU)
    net = net.to(device)
    print(f"Number of parameters : {sum(p.numel() for p in net.parameters())}")
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    train(net, optimizer, trainloader, epochs)#, writer, epochs)
    test_acc = test(net, testloader)
    print(f'Test accuracy:{test_acc}')
    torch.save(net.state_dict(), "mnist_net.pth")


    truetestdata = evaluate(net, truetestloader, index_range=[6774,8462])

    truetestdata.to_csv('truetest_pred_eigen.csv', index=False)

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

