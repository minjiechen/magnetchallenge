# coding=utf-8
import torch
from network import img_network
import torch.nn.functional as F
import numpy as np


def accuracy(network, loader):
    loss_array = []
    network.eval()
    p_len = network.patch_len
    with torch.no_grad():
        for data in loader:
            all_b_ = data[0].cuda().float()
            all_freq_ = data[1].cuda().float()
            all_t_ = data[2].cuda().float()
            all_pk_ = data[3].cuda().float()

            all_h_ = data[4].cuda().float()  # 和norm的h比较, 预测出的h也是norm后的h

            all_h_noise_ = (torch.rand(all_b_.size()[0], all_b_.size()[1] + p_len, 1) * 2 - 1).to(
                torch.device("cuda"))  # (batch_size, seq_len + patch_len, 1) * 2 - 1
            # all_h_noise_[:, 0:16, :] = 0.1 * torch.ones(all_h_noise_[:, 0:16, :].size())
            all_h_noise_[:, 0:p_len, :] = 0.1 * torch.ones(all_h_noise_[:, 0:p_len, :].size())

            # patch_num = all_b_.size()[1] // 16 + 1
            patch_num = all_b_.size()[1] // p_len + 1

            for t in range(1, patch_num):
                p = network.predict(all_b_, all_h_noise_, all_freq_, all_t_, all_pk_)
                # all_h_noise_[:, t * 16: (t + 1) * 16, :] = p[:, (t - 1) * 16: t * 16, :]
                all_h_noise_[:, t * p_len: (t + 1) * p_len, :] = p[:, (t - 1) * p_len: t * p_len, :]

            final_p = network.predict(all_b_, all_h_noise_, all_freq_, all_t_, all_pk_)

            regressor_loss = F.mse_loss(final_p, all_h_)
            loss_array.append(regressor_loss.item())
            # print(f"now loss array len {len(loss_array)}")

    network.train()
    return np.array(loss_array)


def predict_transform(network, loader, norm_dict):
    loss_array = []
    network.eval()
    mean_H = torch.from_numpy(norm_dict["mean_H"]).to(torch.device("cuda"))
    std_H = torch.from_numpy(norm_dict["std_H"]).to(torch.device("cuda"))
    p_len = network.patch_len
    # mean_freq = 10 ** torch.from_numpy(norm_dict["mean_freq_log10"]).to(torch.device("cuda"))
    # std_Freq = 10 ** torch.from_numpy(norm_dict["std_freq_log10"]).to(torch.device("cuda"))
    with torch.no_grad():
        for data in loader:
            all_b_ = data[0].cuda().float()
            all_freq_ = data[1].cuda().float()
            all_t_ = data[2].cuda().float()
            all_pk_ = data[3].cuda().float()

            # all_h_ = data[4].cuda().float()
            # all_ori_h = data[7].cuda().float()
            # all_ori_loss = data[9].cuda().float()

            all_ori_b = data[6].cuda().float()
            all_ori_freq = data[8].cuda().float()

            all_h_noise_ = (torch.rand(all_b_.size()[0], all_b_.size()[1] + p_len, 1) * 2 - 1).to(
                torch.device("cuda"))  # 16为一个patch
            all_h_noise_[:, 0:p_len, :] = 0.1 * torch.ones(all_h_noise_[:, 0:p_len, :].size())

            patch_num = all_b_.size()[1] // p_len + 1

            for t in range(1, patch_num):
                p = network.predict(all_b_, all_h_noise_, all_freq_, all_t_, all_pk_)
                all_h_noise_[:, t * p_len: (t + 1) * p_len, :] = p[:, (t - 1) * p_len: t * p_len, :]

            final_p = network.predict(all_b_, all_h_noise_, all_freq_, all_t_, all_pk_)  # batch * seq_len * 1

            final_p_transform = final_p * std_H + mean_H

            loss = all_ori_freq * torch.trapz(final_p_transform, all_ori_b, dim=1)

            # loss_array.append(final_p_transform)
            loss_array.append(loss)
            # print(f"now loss array len {len(loss_array)}")

    network.train()
    return loss_array

# def predict_transform_abs(network, loader, norm_dict):
#     loss_array = []
#     network.eval()
#     mean_H = torch.from_numpy(norm_dict["mean_H"]).to(torch.device("cuda"))
#     std_H = torch.from_numpy(norm_dict["std_H"]).to(torch.device("cuda"))
#     # mean_freq = 10 ** torch.from_numpy(norm_dict["mean_freq_log10"]).to(torch.device("cuda"))
#     # std_Freq = 10 ** torch.from_numpy(norm_dict["std_freq_log10"]).to(torch.device("cuda"))
#     with torch.no_grad():
#         for data in loader:
#             all_b_ = data[6].cuda().float()
#             all_h_ = data[7].cuda().float()
#             all_t_ = data[8].cuda().float()
#             all_freq_ = data[9].cuda().float()
#             all_ori_b = data[0].cuda().float()
#             all_ori_freq = data[2].cuda().float()
#             all_h_noise_ = (torch.rand(all_b_.size()[0], all_b_.size()[1] + 16, 1) * 2 - 1).to(
#                 torch.device("cuda"))  # 16为一个patch
#             all_h_noise_[:, 0:16, :] = 0.1 * torch.ones(all_h_noise_[:, 0:16, :].size())
#
#             patch_num = all_b_.size()[1] // 16 + 1
#
#             for t in range(1, patch_num):
#                 p = network.predict(all_b_, all_h_noise_, all_freq_, all_t_)
#                 all_h_noise_[:, t * 16: (t + 1) * 16, :] = p[:, (t - 1) * 16: t * 16, :]
#
#             final_p = network.predict(all_b_, all_h_noise_, all_freq_, all_t_)  # batch * seq_len * 1
#
#             final_p_transform = final_p * std_H + mean_H
#
#             loss = all_ori_freq * torch.trapz(final_p_transform, all_ori_b, dim=1)
#
#             # loss_array.append(final_p_transform)
#             loss_array.append(loss.abs())
#             # print(f"now loss array len {len(loss_array)}")
#
#     network.train()
#     return loss_array
