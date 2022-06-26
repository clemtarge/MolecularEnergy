
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(31, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc25 = nn.Linear(200, 400)
        self.fc26 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 1)
        # self.fc1 = nn.Linear(529, 1024)
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, 200)
        # self.fc4 = nn.Linear(200, 100)
        # self.fc5 = nn.Linear(100, 1)

    def forward(self, pos, atom_number, n_atom):
        # positions_cat = F.relu(self.fc1(positions_cat))
        # # positions_cat = F.dropout(positions_cat, 0.5) #dropout was included to combat overfitting
        # positions_cat = F.relu(self.fc2(positions_cat))
        # # positions_cat = F.dropout(positions_cat, 0.5)
        # positions_cat = F.relu(self.fc3(positions_cat))
        # # positions_cat = F.dropout(positions_cat, 0.5)
        # positions_cat = F.relu(self.fc4(positions_cat))
        # positions_cat = F.dropout(positions_cat, 0.5)
        positions_cat = torch.cat((pos, atom_number, n_atom),2)
        positions_cat = F.relu(self.fc1(positions_cat))
        #positions_cat = F.dropout(positions_cat, 0.2)
        positions_cat = F.relu(self.fc2(positions_cat))
        # positions_cat = F.relu(self.fc25(positions_cat))
        # positions_cat = F.relu(self.fc26(positions_cat))
        #positions_cat = F.dropout(positions_cat, 0.2)
        positions_cat = F.relu(self.fc3(positions_cat))
        #positions_cat = F.dropout(positions_cat, 0.2)
        positions_cat = self.fc4(positions_cat)
        return positions_cat


if __name__ == '__main__':
    x = torch.rand(1, 15, 3)
    net = Conv()
    y = net(x)
