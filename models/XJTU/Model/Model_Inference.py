import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

Material = "Material E"

pretrain_model_path = "../train model/" + Material


input_size = 16
hidden_size = 32
num_layers = 1
output_size = 1
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size= input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size= output_size
        self.num_directions = 2 
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=25),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=10)
            )
        # 输入通道数in_channels，输出通道数out_channels，卷积核大小25
        # 池化层最大池化，卷积核大小25，移动步长1
        # 卷积层输入为（batch_size, feature or out_channels, seq_len）
        # 卷积层输出为（batch_size, feature or out_channels, seq_len）
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Sequential(
            nn.Linear(self.num_directions * self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 13)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
            )


    def forward(self, x, y, z, w):
        x = x.permute(0, 2, 1)   # 第二维度和第三维度交换位置
        x = self.conv(x)
        x = x.permute(0, 2, 1)   # 第二维度和第三维度交换位置
        h_0 = torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size)  # 构造h0
        c_0 = torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:,-1,:]
        out = self.fc1(out)
        out = self.fc2(torch.cat((out,y,z,w),1))
        return out


#%% Load Dataset

def load_dataset(in_file1="../Test_Data/Testing/"+Material+"/B_Field.csv", 
                 in_file2="../Test_Data/Testing/"+Material+"/Frequency.csv", 
                 in_file3="../Test_Data/Testing/"+Material+"/Temperature.csv"):
    
    data_B = np.genfromtxt(in_file1, delimiter=',') # N by 1024, in T
    data_F = np.genfromtxt(in_file2, delimiter=',') # N by 1, in Hz
    data_T = np.genfromtxt(in_file3, delimiter=',') # N by 1, in C

    return data_B, data_F, data_T

def load_train_dataset(in_file1="../Test_Data/Training/"+Material+"/B_Field.csv", 
                 in_file2="../Test_Data/Training/"+Material+"/Frequency.csv", 
                 in_file3="../Test_Data/Training/"+Material+"/Temperature.csv",
                 in_file4="../Test_Data/Training/"+Material+"/Volumetric_Loss.csv"):
    
    train_data_B = np.genfromtxt(in_file1, delimiter=',') # N by 1024, in T
    train_data_F = np.genfromtxt(in_file2, delimiter=',') # N by 1, in Hz
    train_data_T = np.genfromtxt(in_file3, delimiter=',') # N by 1, in C
    train_data_P = np.genfromtxt(in_file4, delimiter=',') # N by 1, in C

    return train_data_B, train_data_F, train_data_T, train_data_P

#%% Calculate Core Loss
def core_loss(data_B, data_F, data_T, train_data_B, train_data_F, train_data_T, train_data_P):
    
    #================ Wrap your model or algorithm here=======================#
    
    pre_seq = np.array(train_data_B)
    pre_f = np.array(train_data_F)
    pre_T = np.array(train_data_T)
    pre_lab = np.array(train_data_P)

    pre_f = np.log(pre_f)
    pre_lab = np.log(pre_lab)

    pre_Bm=[]
    for i in range(pre_seq.shape[0]):
        pre_Bm.append(np.max(pre_seq[i,:]))
    pre_Bm = np.log(pre_Bm)

    pre_seq_max = np.max(pre_seq)
    pre_seq_min = np.min(pre_seq)
    pre_f_max = np.max(pre_f)
    pre_f_min = np.min(pre_f)
    pre_T_max = np.max(pre_T)
    pre_T_min = np.min(pre_T)
    pre_lab_max = np.max(pre_lab)
    pre_lab_min = np.min(pre_lab)
    pre_Bm_max = np.max(pre_Bm)
    pre_Bm_min = np.min(pre_Bm)


    seq = np.array(data_B)
    f = np.array(data_F)
    T = np.array(data_T)

    f = np.log(f)

    Bm=[]
    for i in range(seq.shape[0]):
        Bm.append(np.max(seq[i,:]))
    Bm = np.log(Bm)

    data_size = f.shape[0]

    seq_normalized = (2 * (seq.reshape(-1,1) - pre_seq_min)/(pre_seq_max - pre_seq_min) - 1).reshape(data_size,1024)
    f_normalized = (2 * (f.reshape(-1,1) - pre_f_min)/(pre_f_max - pre_f_min) - 1).reshape(data_size,1)
    T_normalized = (2 * (T.reshape(-1,1) - pre_T_min)/(pre_T_max - pre_T_min) - 1).reshape(data_size,1)
    Bm_normalized = (2 * (Bm.reshape(-1,1) - pre_Bm_min)/(pre_Bm_max - pre_Bm_min) - 1).reshape(data_size,1)

    

    valid_seq = torch.FloatTensor(seq_normalized)
    valid_f = torch.FloatTensor(f_normalized)
    valid_T = torch.FloatTensor(T_normalized)
    valid_Bm = torch.FloatTensor(Bm_normalized)

    valid_seq = valid_seq.unsqueeze(2)

    model = LSTM(input_size, hidden_size, num_layers, output_size)  # 创建LSTM类对象
    net = torch.load(pretrain_model_path+"/model.pth")
    model.load_state_dict(net)

    model.eval()
    with torch.no_grad():
        pred = model(valid_seq,valid_f,valid_T,valid_Bm)


    pred = ((pred.reshape(-1,1) + 1)/2)*(pre_lab_max - pre_lab_min) + pre_lab_min


    data_P = np.exp(pred)


#    data_P = pd.DataFrame(pred)
#    data_P.to_csv('../Result/Volumetric_Loss_' + Material +'.csv',index = False, header=None)

    
    #=========================================================================#
    
    with open("../Result/Volumetric_Loss_"+Material+".csv", "w") as file:
        np.savetxt(file, data_P)
        file.close()

    print('Model inference is finished!')
    
    return

#%% Main Function for Model Inference

def main():
    
    # Reproducibility
    random.seed(1)
    np.random.seed(1)

    data_B, data_F, data_T = load_dataset()
    train_data_B, train_data_F, train_data_T, train_data_P = load_train_dataset()
    
    core_loss(data_B, data_F, data_T, train_data_B, train_data_F, train_data_T, train_data_P)
    
if __name__ == "__main__":
    main()
    
# End
