# coding=utf-8
import time

from torch.utils.data import Dataset
import torch
import numpy as np
from datautil.util import Nmax
from utils.util import init_norm_dict
from PIL import Image
from datautil.imgdata.util import rgb_loader, l_loader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
# from torchvision.datasets.folder import npz_loader
import os


# 初始化瓶颈
def SeriesFolder(folder_path):
    List_tuple = []
    for files in os.listdir(folder_path):
        file_path = os.path.join(folder_path, files)
        _ = np.load(file_path)["data"]
        __ = _[-1]
        List_tuple.append((file_path, __))
    return List_tuple


def norm_std(value, mean, std):
    return (value - mean) / std


def load_dataset(dataset_root="MagNet 2023 Database", material=None, mode=None):
    start = time.time()
    if mode == "train":
        b_file = os.path.join(dataset_root, material, "B_waveform[T].csv")
        freq_file = os.path.join(dataset_root, material, "Frequency[Hz].csv")
        t_file = os.path.join(dataset_root, material, "Temperature[C].csv")
        loss_file = os.path.join(dataset_root, material, "Volumetric_losses[Wm-3].csv")
        h_file = os.path.join(dataset_root, material, "H_waveform[Am-1].csv")
    elif mode == "final":
        b_file = os.path.join(dataset_root, material, "B_Field.csv")
        freq_file = os.path.join(dataset_root, material, "Frequency.csv")
        t_file = os.path.join(dataset_root, material, "Temperature.csv")
        loss_file = os.path.join(dataset_root, material, "Volumetric_Loss.csv")
        h_file = os.path.join(dataset_root, material, "H_Field.csv")
    # elif mode == "final_test":
    #     b_file = os.path.join(dataset_root, material, "B_Field.csv")
    #     freq_file = os.path.join(dataset_root, material, "Frequency.csv")
    #     t_file = os.path.join(dataset_root, material, "Temperature.csv")
    #     loss_file = os.path.join(dataset_root, material, "Volumetric_Loss.csv")
    #     h_file = os.path.join(dataset_root, material, "H_Field.csv")
    else:
        b_file = os.path.join(dataset_root, material, "B_waveform.csv")
        freq_file = os.path.join(dataset_root, material, "Frequency.csv")
        t_file = os.path.join(dataset_root, material, "Temperature.csv")
        loss_file = os.path.join(dataset_root, material, "Volumetric_Loss.csv")  # 为啥文件夹名字不format啊
        h_file = os.path.join(dataset_root, material, "H_Waveform.csv")  # 为啥文件夹名字不format啊

    data_b = np.genfromtxt(b_file, delimiter=',')
    data_f = np.genfromtxt(freq_file, delimiter=',')
    data_t = np.genfromtxt(t_file, delimiter=',')
    data_h = np.genfromtxt(h_file, delimiter=',')
    data_loss = np.genfromtxt(loss_file, delimiter=',')
    print(f"load {material}.csv cost {time.time() - start}s")

    return data_b, data_f, data_t, data_h, data_loss


def load_dataset_final_eval(dataset_root="MagNet 2023 Database", material=None):
    start = time.time()

    b_file = os.path.join(dataset_root, material, "B_Field.csv")
    freq_file = os.path.join(dataset_root, material, "Frequency.csv")
    t_file = os.path.join(dataset_root, material, "Temperature.csv")

    data_b = np.genfromtxt(b_file, delimiter=',')
    data_f = np.genfromtxt(freq_file, delimiter=',')
    data_t = np.genfromtxt(t_file, delimiter=',')

    print(f"load {material}.csv cost {time.time() - start}s")

    return data_b, data_f, data_t


def load_dataset_final_eval_func(b_path, t_path, f_path, material=None):
    start = time.time()

    data_b = np.genfromtxt(b_path, delimiter=',')
    data_f = np.genfromtxt(f_path, delimiter=',')
    data_t = np.genfromtxt(t_path, delimiter=',')

    print(f"load {material}.csv cost {time.time() - start}s")

    return data_b, data_f, data_t


class SeriesDatasetSimple(Dataset):
    def __init__(self, dataset, task, root_dir, domain_name, domain_label=-1, labels=None, transform=None,
                 target_transform=None, indices=None, test_envs=[], mode='Default', norm_dict=None):
        # self.datas = SeriesFolder("D:\PycharmProjects\\bh_loss\domain_data_npz\\3C90")
        path = os.path.join(root_dir, "valid_data_npz", domain_name)
        print(path)
        self.datas = SeriesFolder(path)
        self.domain_num = 0
        self.task = task
        self.dataset = dataset
        imgs = [item[0] for item in self.datas]
        labels = [item[1] for item in self.datas]
        self.labels = np.array(labels)
        self.x = imgs
        self.target_transform = target_transform
        if indices is None:
            self.indices = np.arange(len(imgs))
        else:
            self.indices = indices
        self.loader = npz_loader
        self.dlabels = np.ones(self.labels.shape) * (domain_label - Nmax(test_envs, domain_label))
        self.norm_dict = norm_dict

    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def __getitem__(self, index):
        index = self.indices[index]
        img_ = self.loader(self.x[index])
        freq_ = np.array(img_[0]).reshape(1)
        freq_log10 = np.log10(freq_)
        t_ = np.array(img_[1]).reshape(1)
        b_ = img_[2:1026][:, np.newaxis]  # 提取B值
        h_ = img_[1026:2050][:, np.newaxis]  # 提取H值
        bpkpk = torch.tensor(b_.max() - b_.min()).reshape(1, 1)

        freq = torch.tensor(freq_)
        b = torch.tensor(b_)
        h = torch.tensor(h_)

        norm_b = (b_ - self.norm_dict["mean_B"]) / self.norm_dict["std_B"]
        norm_h = (h_ - self.norm_dict["mean_H"]) / self.norm_dict["std_H"]
        norm_t = (t_ - self.norm_dict["mean_T"]) / self.norm_dict["std_T"]
        norm_freq_log10 = (freq_log10 - self.norm_dict["mean_freq_log10"]) / self.norm_dict["std_freq_log10"]

        norm_b_nt = torch.tensor(norm_b)
        norm_h_nt = torch.tensor(norm_h)
        norm_t_nt = torch.tensor(norm_t).reshape(1, )
        norm_freq_nt = torch.tensor(norm_freq_log10).reshape(1, )

        # 如果要在dataloader阶段就构建noise序列的话那么应该加上的大小为一个patch的大小16/32
        #
        # head = 0.1 * torch.ones(h.size()[0], h.size()[1])
        # noise_h = head + (torch.rand(h.size()) - 0.5) * 0.1  # 随机噪声序列H
        # 验证集的h应该是一个随机序列，其实可以在验证时随机生成,写在predict方法里面
        head = 0.1 * torch.ones(16, norm_h_nt.size()[1])
        h_head = torch.cat((head, norm_h_nt), dim=0)
        h_noize = h_head + (torch.rand(h_head.size()) - 0.5) * 0.1

        ctarget = np.array((self.labels[index])).reshape(1)
        ctarget_log10 = np.log10(ctarget)
        dtarget = np.array(self.dlabels[index]).reshape(1)

        return b, h, freq, ctarget, dtarget, norm_b_nt, norm_h_nt, norm_t_nt, norm_freq_nt, bpkpk, h_head, h_noize

    def __len__(self):
        return len(self.indices)


# 不能在初始化的时候全部读,内存会爆炸
class SeriesDatasetFile(Dataset):
    def __init__(self, dataset, task, root_dir, domain_name, domain_label=-1, labels=None, transform=None,
                 target_transform=None, indices=None, test_envs=[], mode=None, norm_dict=None):
        # self.datas = SeriesFolder("D:\PycharmProjects\\bh_loss\domain_data_npz\\3C90")
        # path = os.path.join(root_dir, "valid_data_npz", domain_name)
        # print(path)
        self.datas = load_dataset(dataset_root=root_dir, material=domain_name, mode=mode)
        self.domain_num = 0
        self.task = task
        self.dataset = dataset
        self.labels = np.array(self.datas[4].reshape(-1, 1))
        self.x = self.datas[0:4]
        # self.target_transform = target_transform
        if indices is None:
            self.indices = np.arange(len(self.labels))
        else:
            self.indices = indices
        # self.loader = npz_loader
        self.dlabels = np.ones(self.labels.shape) * (domain_label - Nmax(test_envs, domain_label))
        # self.norm_dict = norm_dict
        self.std_freq = torch.from_numpy(norm_dict["std_freq_log10"])
        self.mean_freq = torch.from_numpy(norm_dict["mean_freq_log10"])

        self.std_T = torch.from_numpy(norm_dict["std_T"])
        self.mean_T = torch.from_numpy(norm_dict["mean_T"])

        self.std_B = torch.from_numpy(norm_dict["std_B"])
        self.mean_B = torch.from_numpy(norm_dict["mean_B"])

        self.std_H = torch.from_numpy(norm_dict["std_H"])
        self.mean_H = torch.from_numpy(norm_dict["mean_H"])

        self.in_b = torch.from_numpy(self.x[0]).float().view(-1, 1024, 1)
        self.in_f = torch.from_numpy(self.x[1]).float().view(-1, 1)
        self.in_t = torch.from_numpy(self.x[2]).float().view(-1, 1)
        self.in_h = torch.from_numpy(self.x[3]).float().view(-1, 1024, 1)

        # self.labels is loss

    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def __getitem__(self, index):

        index = self.indices[index]

        img_b = self.in_b[index]
        img_h = self.in_h[index]
        img_f = self.in_f[index]
        img_t = self.in_t[index]

        norm_b = norm_std(img_b, self.mean_B, self.std_B).reshape(1024, 1)
        norm_f = norm_std(torch.log10(img_f), self.mean_freq, self.std_freq)
        norm_t = norm_std(img_t, self.mean_T, self.std_T)
        norm_h = norm_std(img_h, self.mean_H, self.std_H).reshape(1024, 1)

        b_max = norm_b.clone().detach().max(0)[0]
        b_min = norm_b.clone().detach().min(0)[0]
        bpkpk = (b_max - b_min).reshape(-1, 1)

        # 如果要在dataloader阶段就构建noise序列的话那么应该加上的大小为一个patch的大小16/32
        #
        # head = 0.1 * torch.ones(h.size()[0], h.size()[1])
        # noise_h = head + (torch.rand(h.size()) - 0.5) * 0.1  # 随机噪声序列H
        # 验证集的h应该是一个随机序列，其实可以在验证时随机生成,写在predict方法里面
        # head = 0.1 * torch.ones(norm_h.size()[0], 1, norm_h.size()[1])
        head = 0.1 * torch.ones(8, norm_h.size()[1])  # head的大小和patch的大小相关,是patch的第0的维度
        # norm_h_head_p = torch.cat((head, norm_h), dim=1)
        norm_h_head_p = torch.cat((head, norm_h), dim=0)
        norm_h_head = norm_h_head_p + (torch.rand(norm_h_head_p.size()) - 0.5) * 0.1

        ctarget = np.array((self.labels[index])).reshape(-1, )  # self.label
        dtarget = np.array(self.dlabels[index]).reshape(-1, )

        return norm_b, norm_f, norm_t, bpkpk, norm_h, norm_h_head, img_b, img_h, img_f, ctarget, dtarget

    def __len__(self):
        return len(self.indices)


class SeriesDatasetFileFinal(Dataset):
    def __init__(self, dataset, task, root_dir, domain_name, domain_label=-1, labels=None, transform=None,
                 target_transform=None, indices=None, test_envs=[], mode=None, norm_dict=None):
        # path = os.path.join(root_dir, domain_name)
        # print(path)
        # self.datas = load_dataset(dataset_root=root_dir, material=domain_name, mode=mode)
        self.datas = load_dataset_final_eval(dataset_root=root_dir, material=domain_name)
        self.domain_num = 0
        self.task = task
        self.dataset = dataset
        # self.labels = np.array(self.datas[4].reshape(-1, 1))
        self.labels = np.array(self.datas[2].reshape(-1, 1))
        self.x = self.datas[0:4]
        # self.target_transform = target_transform
        if indices is None:
            self.indices = np.arange(len(self.labels))
        else:
            self.indices = indices
        # self.loader = npz_loader
        self.dlabels = np.ones(self.labels.shape) * (domain_label - Nmax(test_envs, domain_label))
        # self.norm_dict = norm_dict
        self.std_freq = torch.from_numpy(norm_dict["std_freq_log10"])
        self.mean_freq = torch.from_numpy(norm_dict["mean_freq_log10"])

        self.std_T = torch.from_numpy(norm_dict["std_T"])
        self.mean_T = torch.from_numpy(norm_dict["mean_T"])

        self.std_B = torch.from_numpy(norm_dict["std_B"])
        self.mean_B = torch.from_numpy(norm_dict["mean_B"])

        self.std_H = torch.from_numpy(norm_dict["std_H"])
        self.mean_H = torch.from_numpy(norm_dict["mean_H"])

        self.in_b = torch.from_numpy(self.x[0]).float().view(-1, 1024, 1)
        self.in_f = torch.from_numpy(self.x[1]).float().view(-1, 1)
        self.in_t = torch.from_numpy(self.x[2]).float().view(-1, 1)
        # self.in_h = torch.from_numpy(self.x[3]).float().view(-1, 1024, 1)

        # self.labels is loss

    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def __getitem__(self, index):

        index = self.indices[index]

        img_b = self.in_b[index]
        # img_h = self.in_h[index]
        img_h = torch.zeros_like(img_b)  # pad the h
        img_f = self.in_f[index]
        img_t = self.in_t[index]

        norm_b = norm_std(img_b, self.mean_B, self.std_B).reshape(1024, 1)
        norm_f = norm_std(torch.log10(img_f), self.mean_freq, self.std_freq)
        norm_t = norm_std(img_t, self.mean_T, self.std_T)
        norm_h = norm_std(img_h, self.mean_H, self.std_H).reshape(1024, 1)

        b_max = norm_b.clone().detach().max(0)[0]
        b_min = norm_b.clone().detach().min(0)[0]
        bpkpk = (b_max - b_min).reshape(-1, 1)

        # 如果要在dataloader阶段就构建noise序列的话那么应该加上的大小为一个patch的大小16/32
        #
        # head = 0.1 * torch.ones(h.size()[0], h.size()[1])
        # noise_h = head + (torch.rand(h.size()) - 0.5) * 0.1  # 随机噪声序列H
        # 验证集的h应该是一个随机序列，其实可以在验证时随机生成,写在predict方法里面
        # head = 0.1 * torch.ones(norm_h.size()[0], 1, norm_h.size()[1])
        head = 0.1 * torch.ones(8, norm_h.size()[1])  # head的大小和patch的大小相关,是patch的第0的维度
        # norm_h_head_p = torch.cat((head, norm_h), dim=1)
        norm_h_head_p = torch.cat((head, norm_h), dim=0)
        norm_h_head = norm_h_head_p + (torch.rand(norm_h_head_p.size()) - 0.5) * 0.1

        ctarget = np.array((self.labels[index])).reshape(-1, )  # self.label
        dtarget = np.array(self.dlabels[index]).reshape(-1, )

        return norm_b, norm_f, norm_t, bpkpk, norm_h, norm_h_head, img_b, img_h, img_f, ctarget, dtarget

    def __len__(self):
        return len(self.indices)


class SeriesDatasetFileFinalFunc(Dataset):
    def __init__(self, dataset, task, domain_name, b_path, t_path, f_path, domain_label=-1, indices=None,
                 test_envs=[], norm_dict=None):
        # path = os.path.join(root_dir, domain_name)
        # print(path)
        self.datas = load_dataset_final_eval_func(b_path, t_path, f_path, material=domain_name)
        self.domain_num = 0
        self.task = task
        self.dataset = dataset
        self.labels = np.array(self.datas[2].reshape(-1, 1))
        self.x = self.datas[0:4]
        if indices is None:
            self.indices = np.arange(len(self.labels))
        else:
            self.indices = indices
        # self.loader = npz_loader
        self.dlabels = np.ones(self.labels.shape) * (domain_label - Nmax(test_envs, domain_label))
        self.std_freq = torch.from_numpy(norm_dict["std_freq_log10"])
        self.mean_freq = torch.from_numpy(norm_dict["mean_freq_log10"])

        self.std_T = torch.from_numpy(norm_dict["std_T"])
        self.mean_T = torch.from_numpy(norm_dict["mean_T"])

        self.std_B = torch.from_numpy(norm_dict["std_B"])
        self.mean_B = torch.from_numpy(norm_dict["mean_B"])

        self.std_H = torch.from_numpy(norm_dict["std_H"])
        self.mean_H = torch.from_numpy(norm_dict["mean_H"])

        self.in_b = torch.from_numpy(self.x[0]).float().view(-1, 1024, 1)
        self.in_f = torch.from_numpy(self.x[1]).float().view(-1, 1)
        self.in_t = torch.from_numpy(self.x[2]).float().view(-1, 1)


    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def __getitem__(self, index):

        index = self.indices[index]

        img_b = self.in_b[index]
        # img_h = self.in_h[index]
        img_h = torch.zeros_like(img_b)  # pad the h
        img_f = self.in_f[index]
        img_t = self.in_t[index]

        norm_b = norm_std(img_b, self.mean_B, self.std_B).reshape(1024, 1)
        norm_f = norm_std(torch.log10(img_f), self.mean_freq, self.std_freq)
        norm_t = norm_std(img_t, self.mean_T, self.std_T)
        norm_h = norm_std(img_h, self.mean_H, self.std_H).reshape(1024, 1)

        b_max = norm_b.clone().detach().max(0)[0]
        b_min = norm_b.clone().detach().min(0)[0]
        bpkpk = (b_max - b_min).reshape(-1, 1)

        # 如果要在dataloader阶段就构建noise序列的话那么应该加上的大小为一个patch的大小16/32
        #
        # head = 0.1 * torch.ones(h.size()[0], h.size()[1])
        # noise_h = head + (torch.rand(h.size()) - 0.5) * 0.1  # 随机噪声序列H
        # 验证集的h应该是一个随机序列，其实可以在验证时随机生成,写在predict方法里面
        # head = 0.1 * torch.ones(norm_h.size()[0], 1, norm_h.size()[1])
        head = 0.1 * torch.ones(8, norm_h.size()[1])  # head的大小和patch的大小相关,是patch的第0的维度
        # norm_h_head_p = torch.cat((head, norm_h), dim=1)
        norm_h_head_p = torch.cat((head, norm_h), dim=0)
        norm_h_head = norm_h_head_p + (torch.rand(norm_h_head_p.size()) - 0.5) * 0.1

        ctarget = np.array((self.labels[index])).reshape(-1, )  # self.label
        dtarget = np.array(self.dlabels[index]).reshape(-1, )

        return norm_b, norm_f, norm_t, bpkpk, norm_h, norm_h_head, img_b, img_h, img_f, ctarget, dtarget

    def __len__(self):
        return len(self.indices)


if __name__ == '__main__':
    norm = init_norm_dict("D:\\PycharmProjects\dg-magnet\\var_file\\variables_Material A.npz")
    # testData = SeriesDatasetSimple(dataset='test_dataset', task='test_task',
    #                                root_dir='D:\\PycharmProjects\\dg-magnet\\', domain_name='N27', norm_dict=norm)
    testData = SeriesDatasetFileFinal(dataset='test_dataset', task='test_task',
                                      root_dir='D:\\PycharmProjects\dg-magnet\\2023 MagNet Challenge Testing Data (Public)\\Testing\\',
                                      domain_name='Material A',
                                      norm_dict=norm)

    # a = load_dataset("D:\\PycharmProjects\\dg-magnet\\valid_data", material="3C90")

    # self = testData
    #
    # index = testData.indices[0]
    #
    # index = self.indices[index]
    #
    # img_b = self.in_b[index]
    # img_h = self.in_h[index]
    # img_f = self.in_f[index]
    # img_t = self.in_t[index]
    #
    # norm_b = norm_std(img_b, self.mean_B, self.std_B).reshape(1024, 1)
    # norm_f = norm_std(torch.log10(img_f), self.mean_freq, self.std_freq)
    # norm_t = norm_std(img_t, self.mean_T, self.std_T)
    # norm_h = norm_std(img_h, self.mean_H, self.std_H).reshape(1024, 1)
    #
    # b_max = norm_b.clone().detach().max(0)[0]
    # b_min = norm_b.clone().detach().min(0)[0]
    # bpkpk = (b_max - b_min).reshape(-1, 1)
    #
    # # 如果要在dataloader阶段就构建noise序列的话那么应该加上的大小为一个patch的大小16/32
    # #
    # # head = 0.1 * torch.ones(h.size()[0], h.size()[1])
    # # noise_h = head + (torch.rand(h.size()) - 0.5) * 0.1  # 随机噪声序列H
    # # 验证集的h应该是一个随机序列，其实可以在验证时随机生成,写在predict方法里面
    # # head = 0.1 * torch.ones(norm_h.size()[0], 1, norm_h.size()[1])
    # head = 0.1 * torch.ones(8, norm_h.size()[1])  # head的大小和patch的大小相关,是patch的第0的维度
    # # norm_h_head_p = torch.cat((head, norm_h), dim=1)
    # norm_h_head_p = torch.cat((head, norm_h), dim=0)
    # norm_h_head = norm_h_head_p + (torch.rand(norm_h_head_p.size()) - 0.5) * 0.1
    #
    # ctarget = np.array((self.labels[index])).reshape(-1, )  # self.label
    # dtarget = np.array(self.dlabels[index]).reshape(-1, )
    #
    # # 然后他们用的F的norm是先log10然后再norm，应该norm字典里面已经就是log10后的统计值了
    #
    # aa = testData.__getitem__(0)
    # data_len = testData.__len__()

    # b = a[0].reshape(-1, )
    # h = a[1].reshape(-1, )
    # bh = np.trapz(b, h, axis=1)
