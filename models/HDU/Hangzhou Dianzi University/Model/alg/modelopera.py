# coding=utf-8
import torch
from network import img_network
import torch.nn.functional as F
import numpy as np


def get_fea(args):
    if args.dataset == 'dg5':
        net = img_network.DTNBase()
    elif args.net.startswith('res'):
        net = img_network.ResBase(args.net)
    else:
        net = img_network.VGGBase(args.net)
    return net


def accuracy(network, loader):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = network.predict(x)

            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    network.train()
    return correct / total


# def accuracy(network, loader):
#     correct = 0
#     total = 0
#
#     network.eval()
#     with torch.no_grad():
#         for data in loader:
#             x = data[0].cuda().float()
#             # y = data[1].cuda().long()
#             y = data[1].cuda().float()
#             p = network.predict(x)
#
#             loss = network.loss_function(p, y)
#
#             # if p.size(1) == 1:
#             #     correct += (p.gt(0).eq(y).float()).sum().item()
#             # else:
#             #     correct += (p.argmax(1).eq(y).float()).sum().item()
#             total += len(x)
#
#     total = torch.tensor(total)
#     network.train()
#     return loss / total


def accuracy_new(network, loader):
    correct = 0
    total = 0
    batch_template = []
    loss_array = []
    network.eval()
    with torch.no_grad():
        for data in loader:
            # x = data[0].cuda().float()
            # y = data[1].cuda().long()
            # print(data[0].shape[0])
            if data[0].shape[0] == 32:
                batch_template.append(data)

                if len(batch_template) == 9:  # 和实际用于dg的领域相关 此处用于训练的有9个领域 因此为 9 * batch_size = 288

                    all_bh_ = torch.cat([data[6].cuda().float() for data in batch_template])
                    # all_freq_ = torch.cat([data[2].cuda().float() for data in batch_template])
                    all_freq_ = torch.cat([data[9].cuda().float() for data in batch_template])
                    all_t_ = torch.cat([data[3].cuda().float() for data in batch_template])

                    all_y_ = torch.cat([data[7].cuda().float() for data in batch_template])
                    af_ = all_freq_.unsqueeze(0)
                    at_ = all_t_.unsqueeze(0)

                    p = network.predict(all_bh_, af_, at_)

                    p_ = p.squeeze(0)

                    regressor_loss = F.mse_loss(p_, all_y_)  # 所以这个loss是288个样本，最终结果是要除以288的

                    loss_array.append(regressor_loss.item())

                    batch_template = []

            # if p.size(1) == 1:  # 这段要重写一下
            #     correct += (p.gt(0).eq(y).float()).sum().item()
            # else:
            #     correct += (p.argmax(1).eq(y).float()).sum().item()

            # total += len(x)
    network.train()
    # return correct / total
    return np.array(loss_array)


def accuracy_vit(network, loader):
    batch_template = []
    loss_array = []
    network.eval()
    with torch.no_grad():
        for data in loader:
            # a = next(iter(loader))
            # x = data[0].cuda().float()
            # y = data[1].cuda().long()
            # print(data[0].shape[0])
            if data[0].shape[0] == 32:
                batch_template.append(data)

                if len(batch_template) == 9:
                    # all_bh_ = data[6].cuda().float()
                    # all_freq_ = data[9].cuda().float()
                    # all_t_ = data[3].cuda().float()
                    #
                    # all_y_ = data[7].cuda().float()

                    all_bh_ = torch.cat([data[6].cuda().float() for data in batch_template])
                    # all_freq_ = torch.cat([data[2].cuda().float() for data in batch_template])
                    all_freq_ = torch.cat([data[9].cuda().float() for data in batch_template])
                    all_t_ = torch.cat([data[3].cuda().float() for data in batch_template])

                    all_y_ = torch.cat([data[7].cuda().float() for data in batch_template])

                    af_ = all_freq_.unsqueeze(0)
                    at_ = all_t_.unsqueeze(0)

                    p = network.predict(all_bh_, all_freq_, all_t_)

                    # p_ = p.squeeze(0)

                    regressor_loss = F.mse_loss(p, all_y_)  # 所以这个loss是288个样本，最终结果是要除以288的

                    loss_array.append(regressor_loss.item())

                    batch_template = []

            # if p.size(1) == 1:  # 这段要重写一下
            #     correct += (p.gt(0).eq(y).float()).sum().item()
            # else:
            #     correct += (p.argmax(1).eq(y).float()).sum().item()

            # total += len(x)
    network.train()
    # return correct / total
    return np.array(loss_array)


def accuracy_att(network, loader):
    correct = 0
    total = 0
    batch_template = []
    loss_array = []
    network.eval()
    with torch.no_grad():
        for data in loader:
            # x = data[0].cuda().float()
            # y = data[1].cuda().long()
            # print(data[0].shape[0])
            if data[0].shape[0] == 32:
                batch_template.append(data)

                if len(batch_template) == 9:  # 和实际用于dg的领域相关 此处用于训练的有9个领域 因此为 9 * batch_size = 288

                    all_bh_ = torch.cat([data[6].cuda().float() for data in batch_template])
                    all_freq_ = torch.cat([data[2].cuda().float() for data in batch_template])
                    all_t_ = torch.cat([data[3].cuda().float() for data in batch_template])

                    all_y_ = torch.cat([data[7].cuda().float() for data in batch_template])

                    p = network.predict(all_bh_, all_freq_, all_t_)

                    p_ = p.squeeze(0)

                    regressor_loss = F.mse_loss(p_, all_y_)  # 所以这个loss是288个样本，最终结果是要除以288的

                    loss_array.append(regressor_loss.item())

                    batch_template = []

    network.train()
    # return correct / total
    return np.array(loss_array)


def accuracy_vit_simp(network, loader):
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


def accuracy_simp(network, loader):
    loss_array = []
    network.eval()
    with torch.no_grad():
        for data in loader:
            all_bh_ = data[6].cuda().float()
            all_freq_ = data[9].cuda().float()
            all_t_ = data[3].cuda().float()

            all_y_ = data[7].cuda().float()

            abh = all_bh_.reshape(all_bh_.shape[0], -1)

            p = network.predict(abh, all_freq_, all_t_)
            regressor_loss = F.mse_loss(p, all_y_)
            loss_array.append(regressor_loss.item())

    network.train()
    return np.array(loss_array)


def predict_transform(network, loader):
    predict_array = []
    network.eval()
    with torch.no_grad():
        for data in loader:
            all_bh_ = data[6].cuda().float()
            all_freq_ = data[9].cuda().float()
            all_t_ = data[3].cuda().float()

            p = network.predict(all_bh_, all_freq_, all_t_)
            p_t = 10 ** p

            predict_array.append(p_t)

    network.train()
    return predict_array


def error_transform(network, loader):
    error_array = []
    network.eval()
    with torch.no_grad():
        for data in loader:
            all_bh_ = data[6].cuda().float()
            all_freq_ = data[9].cuda().float()
            all_t_ = data[3].cuda().float()

            all_y_n_transformed = data[4].cuda().float()

            p = network.predict(all_bh_, all_freq_, all_t_)
            p_t = 10 ** p
            error = (p_t - all_y_n_transformed) / p_t

            error_array.append(error)

    network.train()
    # return np.array(error_array)
    return error_array


def accuracy_vit_norm(network, loader, norm_dict):
    loss_array = []
    network.eval()
    with torch.no_grad():
        for data in loader:
            all_bh_ = data[6].cuda().float()
            all_freq_ = data[9].cuda().float()
            all_t_ = data[8].cuda().float()

            all_y_ = data[10].cuda().float()

            p = network.predict(all_bh_, all_freq_, all_t_)
            regressor_loss = F.mse_loss(p, all_y_)
            loss_array.append(regressor_loss.item())

    network.train()
    return np.array(loss_array)


def predict_transform_norm(network, loader):
    predict_array = []
    network.eval()
    with torch.no_grad():
        for data in loader:
            all_bh_ = data[6].cuda().float()
            all_freq_ = data[9].cuda().float()
            all_t_ = data[8].cuda().float()

            p = network.predict(all_bh_, all_freq_, all_t_)
            p_t = 10 ** p

            predict_array.append(p_t)

    network.train()
    return predict_array
