import torch
import numpy as np
import pandas as pd
from Networks import AB
from Networks import CNN_Net

Material="Material A"

Modelpath=Material[:8] + "_" + Material[9:]  # 修改子串
save_file_name="./Results/"+Material+".csv"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def Model_initialize():
    global model,net
    model = AB()
    model.eval()
    model.load_state_dict(torch.load(f'./modelsA/LLG_model_{Modelpath}.pth'))  # load model parameters
    # 输出模型A的总参数量
    total_paramsA = count_parameters(model)
    print(f"Sub-model A parameters: {total_paramsA}")
    net = CNN_Net()
    net.eval()
    net.load_state_dict(torch.load(f'./modelsB/CNN_LOSS_{Modelpath}.pth'))  # load model parameters
    # 输出模型B的总参数量
    total_paramsB = count_parameters(net)
    print(f"Sub-model B parameters: {total_paramsB}")
    total_params=total_paramsA+total_paramsB
    print(f"Total Model parameters: {total_params}")

def core_loss(Bt,Fre,Tem): # Function for predicting V_loss
    row, col = Bt.shape  # Reduce columns of Bt from 1024 to 128 points
    if (col == 1024):
        Bt = Bt[:, ::8]  # 128 points
    data1 = Bt * (1e3)
    Fre = Fre * (1e-5)
    Tem = Tem * (1e-1)
    data2 = torch.cat((Fre, Tem), axis=1)
    data1 = data1
    Bt_real = data1[:, ::4]  # 32 points
    data1 = data1.unsqueeze(1)
    H_pre = model(data1, Bt_real)
    H_pre = H_pre.unsqueeze(1)
    net_out = net(data1, H_pre, data2)
    Core_loss=net_out.data*1e3
    result = pd.DataFrame(Core_loss)
    result.to_csv(save_file_name, index=False, header=None)
    print('Model inference is finished')
    return

if __name__ == '__main__':
    Model_initialize() # initial model

    #Load validation set
    dfdata1 = pd.read_csv('./dataset/Testing/'+Material+'/B_Field.csv',header=None)
    dfdata2 = pd.read_csv('./dataset/Testing/'+Material+'/Frequency.csv',header=None)
    dfdata3 = pd.read_csv('./dataset/Testing/'+Material+'/Temperature.csv',header=None)
    Bt  = torch.FloatTensor(dfdata1.iloc[:, 0:1024].values)
    Fre = torch.FloatTensor(dfdata2.iloc[:,:].values)
    Tem = torch.FloatTensor(dfdata3.iloc[:,:].values)

    core_loss(Bt,Fre,Tem)# predict Core loss and output csv
