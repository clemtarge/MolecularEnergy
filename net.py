
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.max_atom = 24

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(72, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, positions, atomic_number):
        n = positions.shape[1]  # height
        positions = F.pad(positions, (0, 0, 0, self.mapositions_atom - n))
        positions = F.relu(self.conv1(positions))
        positions = F.relu(self.conv2(positions))
        positions = self.flatten(positions)

        positions = F.relu(self.fc1(positions))
        # positions = F.dropout(positions, 0.5) #dropout was included to combat overfitting
        positions = F.relu(self.fc2(positions))
        # positions = F.dropout(positions, 0.5)
        positions = self.fc3(positions)
        return positions


if __name__ == '__main__':
    x = torch.rand(1, 15, 3)
    net = Conv()
    y = net(x, y)
