# coding=utf-8
import torch
from network import img_network
import torch.nn.functional as F
import numpy as np


def accuracy(network, loader):
    loss_array = []
    network.eval()
    with torch.no_grad():
        for data in loader:
            all_bh_ = data[6].cuda().float()
            all_freq_ = data[9].cuda().float()
            all_t_ = data[3].cuda().float()

            all_y_ = data[7].cuda().float()

            p = network.predict(all_bh_, all_freq_, all_t_)
            regressor_loss = F.mse_loss(p, all_y_)
            loss_array.append(regressor_loss.item())

    network.train()
    return np.array(loss_array)


def predict_transform(network, loader):
    loss_array = []
    network.eval()
    with torch.no_grad():
        for data in loader:
            all_bh_ = data[6].cuda().float()
            all_freq_ = data[9].cuda().float()
            all_t_ = data[3].cuda().float()

            all_y_ = data[7].cuda().float()

            p = network.predict(all_bh_, all_freq_, all_t_)
            regressor_loss = F.mse_loss(p, all_y_)
            loss_array.append(regressor_loss.item())

    network.train()
    return np.array(loss_array)