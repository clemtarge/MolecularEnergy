import argparse
from statistics import mean

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import numpy as np

import pickle

import pandas as pd

from net_dinguerize import Conv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
 
    def __init__(self,file_name, start_split = 0, end_split = -1, transform = None, is_test = False):

        with open(file_name, "rb") as file:
            df = pickle.load(file).sort_index()
    
        self.is_test = is_test
        self.x_pos = df.iloc[start_split:end_split,0].values
        if not self.is_test:
            self.y = df.iloc[start_split:end_split,1].values

        min_list = []
        max_list = []
        for pos in self.x_pos:
            min_list.append(np.min(pos))
            max_list.append(np.max(pos))

        if transform is None:
            self.transform = transforms.Normalize(np.min(min_list), np.max(max_list)-np.min(min_list))
        else:
            self.transform = transform
 
    def __len__(self):
        return len(self.x_pos)
   
    def __getitem__(self,idx):
        return_array = [self.transform(torch.tensor([self.x_pos[idx].reshape(23,23)],dtype=torch.float32))]
        if not self.is_test:
            return_array.append(torch.tensor([self.y[idx]],dtype=torch.float32))
            
        #print(return_array[0].shape)
        return return_array

def train(net, optimizer, trainloader, validationloader, epochs=10, patience = None, successive_patience = 1):#, writer, epochs=10):
    criterion = nn.MSELoss()
    validation_loss_array = []
    nb_over_patience = 0
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(trainloader)
        for idx, (x_pos, y) in enumerate(t):
            x_pos, y = x_pos.to(device), y.to(device)
            x_pos = x_pos + 1e-6*torch.randn_like(x_pos)
            outputs = net(x_pos)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx < len(t)-1:
            #t.set_description(f'epoch : {epoch:3d} | training loss: {mean(running_loss):4.6f} | validation loss: {validation_MSE:4.6f} | validation RMSE: {torch.sqrt(validation_MSE):4.6f}')
                t.set_description(f'epoch : {epoch:3d} | training loss: {mean(running_loss):11.6f}')
            else:
                validation_MSE = test(net, validationloader)
                t.set_description(f'epoch : {epoch:3d} | training loss: {mean(running_loss):11.6f} | validation loss: {validation_MSE:11.6f} | validation RMSE: {torch.sqrt(validation_MSE):11.6f}',refresh=True)
            #writer.add_scalar('training loss', mean(running_loss), epoch)
        validation_loss_array.append(validation_MSE)
        if patience is not None and epoch >= patience:
            if validation_loss_array[-1] > validation_loss_array[-patience-1]:
                nb_over_patience += 1
                if nb_over_patience >= successive_patience:
                    break
            else:
                nb_over_patience = 0

def test(model, dataloader):
    cur_MSE = 0
    with torch.no_grad():
        for x_pos, y in dataloader:
            x_pos, y = x_pos.to(device), y.to(device)
            y_hat = model(x_pos)
            cur_MSE += nn.MSELoss()(y_hat, y)

    return cur_MSE/len(dataloader)

def evaluate(model, dataloader, index_range = None):

    y_hat = []
    with torch.no_grad():
        for x_pos in dataloader:
            x_pos = x_pos[0].to(device)
            y_hat += model(x_pos).detach().tolist()

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
    traindata = MyDataset('train_preprocess_coulomb.pkl', end_split=5416)
    testdata = MyDataset('train_preprocess_coulomb.pkl', start_split=5416, end_split=6093, transform=traindata.transform)
    validationdata = MyDataset('train_preprocess_coulomb.pkl', start_split=6093, transform=traindata.transform)
    truetestdata = MyDataset('truetest_preprocess_coulomb.pkl', transform=traindata.transform, is_test=True)

    # dataloaders
    trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True)
    validationloader = DataLoader(validationdata, batch_size=batch_size, shuffle=True)
    truetestloader = DataLoader(truetestdata, batch_size=batch_size, shuffle=False)

    net = Conv()

    # setting net on device(GPU if available, else CPU)
    net = net.to(device)
    print(f"Number of parameters : {sum(p.numel() for p in net.parameters())}")
    #optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

    train(net, optimizer, trainloader, validationloader, epochs)#, patience=100, successive_patience = 5)#, writer, epochs)
    test_acc = test(net, testloader)
    print(f'Test accuracy:{torch.sqrt(test_acc)}')
    torch.save(net.state_dict(), "mnist_net.pth")


    truetestdata = evaluate(net, truetestloader, index_range=[6774,8462])

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

