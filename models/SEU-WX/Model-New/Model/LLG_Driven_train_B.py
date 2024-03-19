import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import os
from torch.utils.data import DataLoader
import shutil
import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Networks import AB
from Networks import CNN_Net
from Networks import myDataset
import torch.nn.init as init

n=128 #1024 to n, samping with the equal distance
batch_size=25
learning_rate=0.0008
epochs=800
log_interval=20

Material="Material A"
Modelpath=Material[:8] + "_" + Material[9:]  # 修改子串
data_dir = './dataset/Training/'+Material+'/'

def init_weights(layer):
    # If it is a convolutional layer, initialize using a normal distribution
    if type(layer) == nn.Conv1d:
        nn.init.normal_(layer.weight, mean=0, std=0.55)
    # If it is a fully connected layer, the weights are initialized using uniform distribution
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.55, b=0.55)
        nn.init.constant_(layer.bias, 0.001)

def main():
    # Load the dataset and preprocess it
    train_dataset= myDataset(data_dir=data_dir)
    for x, element in enumerate(train_dataset):
        if (x == 0):
            Bt = element
        elif (x == 1):
            Fre = element
        elif (x == 3):
            Tem = element
        elif (x == 4):
            V_loss = element

    #Data scaling
    row,col=Bt.shape   # Reduce columns of Bt from 1024 to 128 points
    if(col==1024):
        Bt=Bt[:,::8]   # 128 points
    data1= Bt*(1e3)
    Fre = Fre * (1e-5)
    Tem = Tem * (1e-1)
    data2 = torch.cat((Fre, Tem), axis=1)
    target = V_loss* (1e-3)

    # Data load to torch
    torch_dataset = Data.TensorDataset(data1, data2, target)
    train_data,test_data=random_split(torch_dataset,
                                      [round(0.8*data1.shape[0]),round(0.2*data1.shape[0])],
                                      generator=torch.Generator().manual_seed(43))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=5, shuffle=True)

    model = AB()
    model.eval()
    model.load_state_dict(torch.load(f'./modelsA/LLG_model_{Modelpath}.pth'))  # load model parameters

    net=CNN_Net()
    print(model)
    net.apply(init_weights)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,weight_decay=0.0015)
    #adaptive learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.95)

    k_loss=torch.nn.SmoothL1Loss()

   #open tensorboard showing online
    if os.path.exists('./a_tensorboard_logs/'):
        shutil.rmtree('./a_tensorboard_logs/')
        time.sleep(5)
    writer = SummaryWriter('./a_tensorboard_logs/train_logs_1')
    # tensorboard --logdir='C:/Users/admin/Desktop/挑战赛/code/a_tensorboard_logs/train_logs_1'

    # run the main training loop
    for epoch in range(epochs):
        model.train()
        j = 0
        train_loss = 0
        for batch_idx, (data1, data2, target) in enumerate(train_loader):
            data1 = data1
            Bt_real = data1[:, ::4]  #32 points
            data1 = data1.unsqueeze(1)
            H_pre=model(data1,Bt_real)
            H_pre=H_pre.unsqueeze(1)
            net_out = net(data1, H_pre,data2)
            loss=k_loss(net_out, target)
            train_loss += loss.data
            j=j+1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data1), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data))
                print(target[0], net_out.data[0])
            writer.add_scalar('Loss/train', train_loss/j, epoch)
        if (1200>= epoch >= 40):
            scheduler.step(loss)
        writer.add_scalar('Learning rate/train', np.array(optimizer.param_groups[0]['lr']), epoch)

    model.eval()
    i = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data1, data2, target) in enumerate(test_loader):
            data1 = data1
            Bt_real = data1[:, ::4]  # 32 points
            data1 = data1.unsqueeze(1)
            H_pre = model(data1, Bt_real)
            H_pre = H_pre.unsqueeze(1)
            net_out = net(data1, H_pre, data2)
            # sum up batch loss
            t_loss = k_loss(net_out, target)
            test_loss += t_loss.data
            i=i+1
        print('Loss/test', test_loss/i)
    torch.save(net.state_dict(), f"./modelsB/CNN_LOSS_{Modelpath}.pth")

if __name__ == "__main__":
    main()