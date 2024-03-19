import torch
import torch.nn as nn
import os
import pandas as pd
from torch.utils.data import Dataset

class myDataset(Dataset):

    def __init__(self, data_dir):
        """
        data_dir: Data file path
        """
        # Read the name of each data file in the folder
        self.file_name = os.listdir(data_dir)

        self.data_path = []
        # Splice the paths of each file together
        for index in range(len(self.file_name)):
            self.data_path.append(os.path.join(data_dir, self.file_name[index]))

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, index):
        # load data
        data = pd.read_csv(self.data_path[index], header=None)
        # Convert to Tensor
        data = torch.FloatTensor(data.values)

        return data

class AB(nn.Module):
    def __init__(self):
        super(AB,self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )
        self.layer1 = nn.Sequential(nn.Linear(30, 30), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(30, 20), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(20, 3))

        self.layer21 = nn.Sequential(nn.Linear(35, 128), nn.ReLU(True))
        self.layer22 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(True))
        self.layer23 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(True))
        #self.layer24 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(True))
        self.layer25 = nn.Sequential(nn.Linear(128, 32))

    def forward(self, x, y):
        x = self.con1(x)
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x2 = torch.cat((x, y), 1)
        x2 = self.layer21(x2)
        x2 = self.layer22(x2)
        x2 = self.layer23(x2)
        #x2 = self.layer24(x2)
        x2 = self.layer25(x2)
        return x2

class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )
        self.con2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(4, 4)
        )
        self.fc1 = nn.Sequential(nn.Linear(7, 20), nn.ReLU(True))
        self.fc2 = nn.Sequential(nn.Linear(20, 4),nn.ReLU(True))

        self.layer1 = nn.Sequential(nn.Linear( 36, 150), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(150, 150), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(150, 150), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(150, 150), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(150, 150), nn.ReLU(True))
        #self.layer51 = nn.Sequential(nn.Linear(150, 150), nn.ReLU(True))
        #self.layer52 = nn.Sequential(nn.Linear(100, 100), nn.ReLU(True))
        self.layer6 = nn.Sequential(nn.Linear(150, 1))

    def forward(self, x11,x12, x2):
        x11 = self.con1(x11)
        x11 = torch.flatten(x11, 1)

        x12 = self.con2(x12)
        x12 = torch.flatten(x12, 1)
        x12 = self.fc1(x12)
        x12 = self.fc2(x12)

        x = torch.cat((x11, x12, x2), 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        #x = self.layer51(x)
        #x = self.layer52(x)
        x = self.layer6(x)
        return x