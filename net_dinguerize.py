
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=15, out_channels=13, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=13, out_channels=11, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(33856, 600)
        self.fc15 = nn.Linear(256, 128)
        self.fc16 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(600, 1)
        # self.fc1 = nn.Linear(529, 1024)
        # self.fc2 = nn.Linear(1024, 512)
        # self.fc3 = nn.Linear(512, 200)
        # self.fc4 = nn.Linear(200, 100)
        # self.fc5 = nn.Linear(100, 1)

    def forward(self, positions):
        positions = F.relu(self.conv1(positions))
        # positions = self.maxpool(positions)
        positions = F.relu(self.conv2(positions))
        # positions = self.maxpool(positions)
        positions = F.relu(self.conv3(positions))
        # positions = self.maxpool(positions)
        #positions = F.relu(self.conv4(positions))
        # positions = self.maxpool(positions)
        # positions = F.relu(self.conv5(positions))
        # positions = self.maxpool(positions)
        # positions = F.relu(self.conv6(positions))
        # positions = self.maxpool(positions)
        positions = self.flatten(positions)

        positions_cat = positions

        # positions_cat = F.relu(self.fc1(positions_cat))
        # # positions_cat = F.dropout(positions_cat, 0.5) #dropout was included to combat overfitting
        # positions_cat = F.relu(self.fc2(positions_cat))
        # # positions_cat = F.dropout(positions_cat, 0.5)
        # positions_cat = F.relu(self.fc3(positions_cat))
        # # positions_cat = F.dropout(positions_cat, 0.5)
        # positions_cat = F.relu(self.fc4(positions_cat))
        # positions_cat = F.dropout(positions_cat, 0.5)
        positions_cat = F.relu(self.fc1(positions_cat))
        # positions_cat = F.relu(self.fc15(positions_cat))
        # positions_cat = F.relu(self.fc16(positions_cat))
        #positions_cat = F.dropout(positions_cat, 0.2)
        positions_cat = self.fc2(positions_cat)
        return positions_cat


if __name__ == '__main__':
    x = torch.rand(1, 15, 3)
    net = Conv()
    y = net(x)
