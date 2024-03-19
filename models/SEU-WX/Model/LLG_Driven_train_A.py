from magtense.micromag import MicromagProblem
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import os
import shutil
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
from Networks import AB
from Networks import myDataset
import torch.nn.init as init

n=128 #1024 to n, samping with the equal distance
batch_size=16
learning_rate=0.0004
epochs=6
log_interval=20

Material="Material C"
Modelpath=Material[:8] + "_" + Material[9:]  # 修改子串
data_dir = './dataset/Training/'+Material+'/'

def llg1(res=[2, 2, 1], m0=1, Ht=0, Ms=8e5, A0=1.3e-13, K0=4e6, alpha=4.42e3):
    mu0 = 4 * np.pi * 1e-7
    grid_L = [4e-9, 4e-9, 1e-9]

    ### Time-dependent solver
    problem_dym = MicromagProblem(res, grid_L, m0=m0_ini, Ms=Ms, A0=A0, K0=K0, alpha=alpha,cuda=0)
    problem_dym.u_ea[:, 0] = 0
    problem_dym.u_ea[:, 1] = 1
    problem_dym.u_ea[:, 2] = 0
    # external fields
    HystDir = np.array([0, 1, 0])
    h_ext_nist = HystDir * Ht

    h_ext_fct = lambda t: np.expand_dims(t > -1, axis=1) * h_ext_nist
    t_dym, M_out, _, _, _, _, _ = problem_dym.run_simulation(4e-9, 10, h_ext_fct, 20)

    M_sq_dym = np.squeeze(M_out, axis=2)
    M_average = np.mean(M_sq_dym[-1], axis=0)
    M_LAST = M_sq_dym[-1]
    mx_ave = M_average[0]
    my_ave = M_average[1]
    mz_ave = M_average[2]
    M = mx_ave * HystDir[0] + my_ave * HystDir[1] + mz_ave * HystDir[2]
    B=mu0*(Ht+M*Ms)
    print(h_ext_nist, B)
    return B,M_LAST

def LLGfun(Ht,A_MS,B_A0,C_K0,Bt_pre):
    global m0_ini
    random_m0 = np.random.rand(4, 3)
    row_magnitudes = np.linalg.norm(random_m0, axis=1)
    normalized_vector = random_m0 / row_magnitudes[:, np.newaxis]
    m0_ini = normalized_vector

    Ht=np.array(Ht).reshape(1,32)

    for ii in range(32):
        print(ii)
        z=Ht[0][ii]*1e3
        det,M_LAST=llg1(Ht=z, m0=m0_ini,Ms=A_MS*1e3, A0=B_A0*(1e-11),K0=C_K0*(4e8))
        m0_ini = M_LAST
        #det = float(det)
        Bt_pre[ii]=det
    #Bt_pre=torch.Tensor(Bt_pre)
    return Bt_pre

input_data = {}
output_data = {}

def get_activation(name):
    def hook(model, input, output):
        input_data[name] = input[0].detach()  # input type is tulple, only has one element, which is the tensor
        output_data[name] = output.detach()  # output type is tensor

    return hook

def main():
    # Load the dataset and preprocess it
    train_dataset= myDataset(data_dir=data_dir)
    for x,element in enumerate(train_dataset):
        if(x==0):
            Bt=element
        elif(x==1):
            Fre=element
        elif(x==2):
            Ht=element
        elif(x==3):
            Tem=element

    row,col=Bt.shape   # Reduce columns of Bt from 1024 to 128 points
    if(col==1024):
        Bt=Bt[:,::8]   # 128 points
        Ht=Ht[:,::8]   # 128 points

    #torch.set_printoptions(precision=8)
    data= Bt*(1e3)
    target = Ht

    # 加载数据集
    torch_dataset=Data.TensorDataset(data, target)
    train_loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)

    net = AB()
    print(net)
    net.layer3[0].register_forward_hook(get_activation('materials'))

    if os.path.exists('./a_tensorboard_logs/'):
        shutil.rmtree('./a_tensorboard_logs/')
        time.sleep(5)
    writer = SummaryWriter('./a_tensorboard_logs/train_logs_1')
    # tensorboard --logdir='C:/Users/admin/Desktop/NeuralNetwork/code/a_tensorboard_logs/train_logs_1'

    # create a stochastic gradient descent optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    Loss = torch.nn.MSELoss()

    # run the main training loop
    data2 = torch.rand((batch_size, 3))
    for epoch in range(epochs):
        for batch_idx,(data, target) in enumerate(train_loader):
            Ht_real = target[:,::4]  #Shrink to 32 ponits
            Bt_real = data[:,::4]    #Shrink to 32 ponits

            if(epoch==4 and batch_idx==1):
                torch.save(net.state_dict(), f"./modelsA/LLG_model_{Modelpath}.pth")
            if(len(data)<batch_size):
                break

            data =data.unsqueeze(1)
            LLG_out = torch.empty(batch_size,32)

            processes = []
            for i in range(batch_size):
                p = multiprocessing.Process(target=LLGfun, args=(Ht_real[i][:],np.array(data2[i][0]),
                           np.array(data2[i][1]),np.array(data2[i][2]),LLG_out[i]))
                processes.append(p)
                p.start()
            # Waiting for all sub-processes to end
            for p in processes:
                p.join()

            LLG_out=LLG_out*1e3

            weight_1 = 0.5
            weight_2 = 0.5
            weighted_1 = torch.mul(torch.tensor(LLG_out), weight_1)
            weighted_2 = torch.mul(torch.tensor(Bt_real), weight_2)
            LLG_out = torch.add(weighted_1, weighted_2)
            LLG_out.requires_grad_(True)

            net_out = net(data,LLG_out)
            data2=output_data['materials']
            loss=Loss(net_out, Ht_real*(1e-1))

            writer.add_scalar('Loss/train', loss, epoch*int(len(train_loader.dataset)/batch_size)+batch_idx)
            writer.add_scalar('MS/train', data2[0][0], epoch*int(len(train_loader.dataset)/batch_size)+batch_idx)
            writer.add_scalar('A0/train', data2[0][1], epoch*int(len(train_loader.dataset)/batch_size)+batch_idx)
            writer.add_scalar('K0/train', data2[0][2], epoch * int(len(train_loader.dataset) / batch_size) + batch_idx)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

if __name__ == "__main__":
    main()