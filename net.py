
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(2328, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, positions, atomic_number):
        positions = F.relu(self.conv1(positions))
        positions = F.relu(self.conv2(positions))
        positions = self.flatten(positions)
        atomic_number = self.flatten(atomic_number)

        positions_cat = torch.cat((positions, atomic_number), 1)

        positions_cat = F.relu(self.fc1(positions_cat))
        # positions = F.dropout(positions, 0.5) #dropout was included to combat overfitting
        positions_cat = F.relu(self.fc2(positions_cat))
        # positions = F.dropout(positions, 0.5)
        positions_cat = self.fc3(positions_cat)
        return positions_cat


if __name__ == '__main__':
    x = torch.rand(1, 15, 3)
    net = Conv()
    y = net(x)
