from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np

import matplotlib.pyplot as plt


def get_dataLoader(file_path, batch_size=64, shuffle=True):

    class MyDataset(Dataset):
        def __init__(self, x_data, y_data):
            self.x_data = torch.tensor(x_data, dtype=torch.float32)
            self.y_data = torch.tensor(y_data, dtype=torch.float32)

        def __len__(self):
            return len(self.x_data)

        def __getitem__(self, idx):
            return self.x_data[idx], self.y_data[idx]

    import Maglib

    magData=Maglib.MagLoader(file_path)

    x_data=np.zeros([magData.b.shape[0],magData.b.shape[1],3])
    x_data[:,:,0]=magData.b
    x_data[:,:,1]=magData.freq
    x_data[:,:,2]=magData.temp

    y_data=magData.loss

    # debugIdx=5000
    # print(x_data[debugIdx,:,2],magData.temp[debugIdx])
    # plt.plot(x_data[debugIdx,:,0])
    # plt.show()


    # Assuming x_data and y_data are your numpy arrays
    dataset = MyDataset(x_data, y_data)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":

    get_dataLoader(r"data\std_78_cycle\78_cycle.mat")

    pass
