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


# def SeriesFolder(folder_path):
#     List_tuple = []
#     for root, dirs, files in os.walk(folder_path):
#         for file_name in files:
#             file_path = os.path.join(root, file_name)
#             _ = np.load(file_path)["data"]
#             __ = _[-1]
#             List_tuple.append((file_path, __))
#             # print(file_path)
#     return List_tuple


class SeriesDatasetLSTMNorm(Dataset):
    def __init__(self, dataset, task, root_dir, domain_name, domain_label=-1, labels=None, transform=None,
                 target_transform=None, indices=None, test_envs=[], mode='Default', norm_dict=None):
        print(root_dir + "domain_data_npz_\\" + domain_name)
        self.datas = SeriesFolder(root_dir + "domain_data_npz_\\" + domain_name)
        self.domain_num = 0
        self.task = task
        self.dataset = dataset
        imgs = [item[0] for item in self.datas]
        labels = [item[1] for item in self.datas]
        self.labels = np.array(labels)
        self.x = imgs
        self.x = imgs
        self.transform = transform
        self.target_transform = target_transform
        if indices is None:
            self.indices = np.arange(len(imgs))
        else:
            self.indices = indices
        self.loader = npz_loader

        self.dlabels = np.ones(self.labels.shape) * \
                       (domain_label - Nmax(test_envs, domain_label))

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

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        index = self.indices[index]
        # img = self.input_trans(self.loader(self.x[index]))
        img_ = self.input_trans(self.loader(self.x[index]))
        freq_ = np.array(img_[0]).reshape(1)
        freq_log10 = np.log10(freq_)
        t_ = np.array(img_[1]).reshape(1)
        b_ = img_[2:1026][:, np.newaxis]  # 提取B值
        h_ = img_[1026:2050][:, np.newaxis]  # 提取H值
        bpkpk = torch.tensor(b_.max() - b_.min()).reshape(1, 1)
        # bh__ = np.concatenate((b_, h_), axis=1)
        # bh_ = np.stack((b_, h_)).reshape((2, 1024))
        # img = torch.tensor(img_)
        freq = torch.tensor(freq_)
        # freq_trans = torch.tensor(freq_log10)
        t = torch.tensor(t_)
        b = torch.tensor(b_)
        h = torch.tensor(h_)
        # bh = torch.tensor(bh_)
        # bh_nt = torch.tensor(bh__)

        norm_b = (b_ - self.norm_dict["mean_B"]) / self.norm_dict["std_B"]
        norm_h = (h_ - self.norm_dict["mean_H"]) / self.norm_dict["std_H"]
        norm_t = (t_ - self.norm_dict["mean_T"]) / self.norm_dict["std_T"]
        norm_freq_log10 = (freq_log10 - self.norm_dict["mean_freq_log10"]) / self.norm_dict["std_freq_log10"]

        norm_bh = np.concatenate((norm_b, norm_h), axis=1)
        norm_bh_nt = torch.tensor(norm_bh)
        norm_t_nt = torch.tensor(norm_t).reshape(1, )
        norm_freq_log10_nt = torch.tensor(norm_freq_log10).reshape(1, )

        ctarget = np.array(self.target_trans(self.labels[index])).reshape(1)
        ctarget_trans = np.log10(ctarget)
        dtarget = np.array(self.target_trans(self.dlabels[index])).reshape(1)
        # return b, h, freq, t, ctarget, dtarget, bh, bh_nt, ctarget_trans, bpkpk
        # return b, h, freq, t, ctarget, dtarget, bh_nt, ctarget_trans, bpkpk, freq_trans, norm_bh_nt, norm_t_nt, norm_freq_log10_nt
        return b, h, freq, t, ctarget, dtarget, norm_bh_nt, bpkpk, norm_t_nt, norm_freq_log10_nt, ctarget_trans

    def __len__(self):
        return len(self.indices)

    def test(self, index):
        index = self.indices[index]
        # img = self.input_trans(self.loader(self.x[index]))
        img_ = self.input_trans(self.loader(self.x[index]))

        return img_


class SeriesDatasetValidNorm(Dataset):
    def __init__(self, dataset, task, root_dir, domain_name, domain_label=-1, labels=None, transform=None,
                 target_transform=None, indices=None, test_envs=[], mode='Default', norm_dict=None):
        # self.datas = SeriesFolder("D:\PycharmProjects\\bh_loss\domain_data_npz\\3C90")
        print(root_dir + "valid_data_npz\\" + domain_name)
        self.datas = SeriesFolder(root_dir + "valid_data_npz\\" + domain_name)
        # self.imgs = ImageFolder(root_dir+domain_name).imgs
        # self.imgs = ImageFolder(root_dir+domain_name).imgs

        self.domain_num = 0
        self.task = task
        self.dataset = dataset
        imgs = [item[0] for item in self.datas]
        labels = [item[1] for item in self.datas]
        self.labels = np.array(labels)
        self.x = imgs
        self.x = imgs
        self.transform = transform
        self.target_transform = target_transform
        if indices is None:
            self.indices = np.arange(len(imgs))
        else:
            self.indices = indices
        self.loader = npz_loader

        self.dlabels = np.ones(self.labels.shape) * \
                       (domain_label - Nmax(test_envs, domain_label))

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

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        index = self.indices[index]
        # img = self.input_trans(self.loader(self.x[index]))
        img_ = self.input_trans(self.loader(self.x[index]))
        freq_ = np.array(img_[0]).reshape(1)
        freq_log10 = np.log10(freq_)
        t_ = np.array(img_[1]).reshape(1)
        b_ = img_[2:1026][:, np.newaxis]  # 提取B值
        h_ = img_[1026:2050][:, np.newaxis]  # 提取H值
        bpkpk = torch.tensor(b_.max() - b_.min()).reshape(1, 1)
        # bh__ = np.concatenate((b_, h_), axis=1)
        # bh_ = np.stack((b_, h_)).reshape((2, 1024))
        # img = torch.tensor(img_)
        freq = torch.tensor(freq_)
        # freq_trans = torch.tensor(freq_log10)
        t = torch.tensor(t_)
        b = torch.tensor(b_)
        h = torch.tensor(h_)
        # bh = torch.tensor(bh_)
        # bh_nt = torch.tensor(bh__)

        norm_b = (b_ - self.norm_dict["mean_B"]) / self.norm_dict["std_B"]
        norm_h = (h_ - self.norm_dict["mean_H"]) / self.norm_dict["std_H"]
        norm_t = (t_ - self.norm_dict["mean_T"]) / self.norm_dict["std_T"]
        norm_freq_log10 = (freq_log10 - self.norm_dict["mean_freq_log10"]) / self.norm_dict["std_freq_log10"]

        norm_bh = np.concatenate((norm_b, norm_h), axis=1)
        norm_bh_nt = torch.tensor(norm_bh)
        norm_t_nt = torch.tensor(norm_t).reshape(1, )
        norm_freq_log10_nt = torch.tensor(norm_freq_log10).reshape(1, )

        ctarget = np.array(self.target_trans(self.labels[index])).reshape(1)
        ctarget_trans = np.log10(ctarget)
        dtarget = np.array(self.target_trans(self.dlabels[index])).reshape(1)
        # return b, h, freq, t, ctarget, dtarget, bh, bh_nt, ctarget_trans, bpkpk
        # return b, h, freq, t, ctarget, dtarget, bh_nt, ctarget_trans, bpkpk, freq_trans, norm_bh_nt, norm_t_nt, norm_freq_log10_nt
        return b, h, freq, t, ctarget, dtarget, norm_bh_nt, bpkpk, norm_t_nt, norm_freq_log10_nt, ctarget_trans

    def __len__(self):
        return len(self.indices)

    def test(self, index):
        index = self.indices[index]
        # img = self.input_trans(self.loader(self.x[index]))
        img_ = self.input_trans(self.loader(self.x[index]))

        return img_


class SeriesDatasetLSTMNormb(Dataset):
    def __init__(self, dataset, task, root_dir, domain_name, domain_label=-1, labels=None, transform=None,
                 target_transform=None, indices=None, test_envs=[], mode='Default', norm_dict=None):
        # self.datas = SeriesFolder("D:\PycharmProjects\\bh_loss\domain_data_npz\\3C90")
        print(root_dir + "domain_data_npz_\\" + domain_name)
        self.datas = SeriesFolder(root_dir + "domain_data_npz_\\" + domain_name)
        # self.imgs = ImageFolder(root_dir+domain_name).imgs
        # self.imgs = ImageFolder(root_dir+domain_name).imgs

        self.domain_num = 0
        self.task = task
        self.dataset = dataset
        imgs = [item[0] for item in self.datas]
        labels = [item[1] for item in self.datas]
        self.labels = np.array(labels)
        self.x = imgs
        self.transform = transform
        self.target_transform = target_transform
        if indices is None:
            self.indices = np.arange(len(imgs))
        else:
            self.indices = indices
        self.loader = npz_loader

        self.dlabels = np.ones(self.labels.shape) * \
                       (domain_label - Nmax(test_envs, domain_label))

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

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        index = self.indices[index]
        # img = self.input_trans(self.loader(self.x[index]))
        img_ = self.input_trans(self.loader(self.x[index]))
        freq_ = np.array(img_[0]).reshape(1)
        freq_log10 = np.log10(freq_)
        t_ = np.array(img_[1]).reshape(1)
        b_ = img_[2:1026][:, np.newaxis]  # 提取B值
        h_ = img_[1026:2050][:, np.newaxis]  # 提取H值
        bpkpk = torch.tensor(b_.max() - b_.min()).reshape(1, 1)  # pkpk如果要用肯定也要norm过
        # bh__ = np.concatenate((b_, h_), axis=1)
        # bh_ = np.stack((b_, h_)).reshape((2, 1024))
        # img = torch.tensor(img_)
        freq = torch.tensor(freq_)
        # freq_trans = torch.tensor(freq_log10)
        t = torch.tensor(t_)
        b = torch.tensor(b_)
        h = torch.tensor(h_)

        norm_b = (b_ - self.norm_dict["mean_B"]) / self.norm_dict["std_B"]
        norm_h = (h_ - self.norm_dict["mean_H"]) / self.norm_dict["std_H"]
        norm_t = (t_ - self.norm_dict["mean_T"]) / self.norm_dict["std_T"]
        norm_freq_log10 = (freq_log10 - self.norm_dict["mean_freq_log10"]) / self.norm_dict["std_freq_log10"]

        norm_b_nt = torch.tensor(norm_b)
        norm_h_nt = torch.tensor(norm_h)
        norm_t_nt = torch.tensor(norm_t).reshape(1, )
        norm_freq_log10_nt = torch.tensor(norm_freq_log10).reshape(1, )

        # 如果要在dataloader阶段就构建noise序列的话那么应该加上的大小为一个patch的大小16/32
        #
        # head = 0.1 * torch.ones(h.size()[0], h.size()[1])
        # noise_h = head + (torch.rand(h.size()) - 0.5) * 0.1  # 随机噪声序列H
        head = 0.1 * torch.ones(16, norm_h_nt.size()[1])
        h_head = torch.cat((head, norm_h_nt), dim=0)
        h_noize = h_head + (torch.rand(h_head.size()) - 0.5) * 0.1

        ctarget = np.array(self.target_trans(self.labels[index])).reshape(1)
        ctarget_log10 = np.log10(ctarget)
        dtarget = np.array(self.target_trans(self.dlabels[index])).reshape(1)
        # return b, h, freq, t, ctarget, dtarget, bh, bh_nt, ctarget_trans, bpkpk
        # return b, h, freq, t, ctarget, dtarget, bh_nt, ctarget_trans, bpkpk, freq_trans, norm_bh_nt, norm_t_nt, norm_freq_log10_nt
        return b, h, freq, t, ctarget, dtarget, norm_b_nt, norm_h_nt, norm_t_nt, norm_freq_log10_nt, bpkpk, h_head, h_noize

    def __len__(self):
        return len(self.indices)

    def test(self, index):
        index = self.indices[index]
        # img = self.input_trans(self.loader(self.x[index]))
        img_ = self.input_trans(self.loader(self.x[index]))

        return img_


class SeriesDatasetValidNormb(Dataset):
    def __init__(self, dataset, task, root_dir, domain_name, domain_label=-1, labels=None, transform=None,
                 target_transform=None, indices=None, test_envs=[], mode='Default', norm_dict=None):
        # self.datas = SeriesFolder("D:\PycharmProjects\\bh_loss\domain_data_npz\\3C90")
        print(root_dir + "valid_data_npz\\" + domain_name)
        self.datas = SeriesFolder(root_dir + "valid_data_npz\\" + domain_name)
        # self.imgs = ImageFolder(root_dir+domain_name).imgs
        # self.imgs = ImageFolder(root_dir+domain_name).imgs

        self.domain_num = 0
        self.task = task
        self.dataset = dataset
        imgs = [item[0] for item in self.datas]
        labels = [item[1] for item in self.datas]
        self.labels = np.array(labels)
        self.x = imgs
        self.x = imgs
        self.transform = transform
        self.target_transform = target_transform
        if indices is None:
            self.indices = np.arange(len(imgs))
        else:
            self.indices = indices
        self.loader = npz_loader

        self.dlabels = np.ones(self.labels.shape) * \
                       (domain_label - Nmax(test_envs, domain_label))

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

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        index = self.indices[index]
        # img = self.input_trans(self.loader(self.x[index]))
        img_ = self.input_trans(self.loader(self.x[index]))
        freq_ = np.array(img_[0]).reshape(1)
        freq_log10 = np.log10(freq_)
        t_ = np.array(img_[1]).reshape(1)
        b_ = img_[2:1026][:, np.newaxis]  # 提取B值
        h_ = img_[1026:2050][:, np.newaxis]  # 提取H值
        bpkpk = torch.tensor(b_.max() - b_.min()).reshape(1, 1)
        # bh__ = np.concatenate((b_, h_), axis=1)
        # bh_ = np.stack((b_, h_)).reshape((2, 1024))
        # img = torch.tensor(img_)
        freq = torch.tensor(freq_)
        # freq_trans = torch.tensor(freq_log10)
        t = torch.tensor(t_)
        b = torch.tensor(b_)
        h = torch.tensor(h_)

        norm_b = (b_ - self.norm_dict["mean_B"]) / self.norm_dict["std_B"]
        norm_h = (h_ - self.norm_dict["mean_H"]) / self.norm_dict["std_H"]
        norm_t = (t_ - self.norm_dict["mean_T"]) / self.norm_dict["std_T"]
        norm_freq_log10 = (freq_log10 - self.norm_dict["mean_freq_log10"]) / self.norm_dict["std_freq_log10"]

        norm_b_nt = torch.tensor(norm_b)
        norm_h_nt = torch.tensor(norm_h)
        norm_t_nt = torch.tensor(norm_t).reshape(1, )
        norm_freq_log10_nt = torch.tensor(norm_freq_log10).reshape(1, )

        # 如果要在dataloader阶段就构建noise序列的话那么应该加上的大小为一个patch的大小16/32
        #
        # head = 0.1 * torch.ones(h.size()[0], h.size()[1])
        # noise_h = head + (torch.rand(h.size()) - 0.5) * 0.1  # 随机噪声序列H
        # 验证集的h应该是一个随机序列，其实可以在验证时随机生成,写在predict方法里面
        head = 0.1 * torch.ones(16, norm_h_nt.size()[1])
        h_head = torch.cat((head, norm_h_nt), dim=0)
        h_noize = h_head + (torch.rand(h_head.size()) - 0.5) * 0.1

        ctarget = np.array(self.target_trans(self.labels[index])).reshape(1)
        ctarget_log10 = np.log10(ctarget)
        dtarget = np.array(self.target_trans(self.dlabels[index])).reshape(1)
        # return b, h, freq, t, ctarget, dtarget, bh, bh_nt, ctarget_trans, bpkpk
        # return b, h, freq, t, ctarget, dtarget, bh_nt, ctarget_trans, bpkpk, freq_trans, norm_bh_nt, norm_t_nt, norm_freq_log10_nt
        return b, h, freq, t, ctarget, dtarget, norm_b_nt, norm_h_nt, norm_t_nt, norm_freq_log10_nt, bpkpk, h_head, h_noize

    def __len__(self):
        return len(self.indices)

    def test(self, index):
        index = self.indices[index]
        # img = self.input_trans(self.loader(self.x[index]))
        img_ = self.input_trans(self.loader(self.x[index]))

        return img_


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

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        index = self.indices[index]
        # img = self.input_trans(self.loader(self.x[index]))
        img_ = self.input_trans(self.loader(self.x[index]))
        freq_ = np.array(img_[0]).reshape(1)
        freq_log10 = np.log10(freq_)
        t_ = np.array(img_[1]).reshape(1)
        b_ = img_[2:1026][:, np.newaxis]  # 提取B值
        h_ = img_[1026:2050][:, np.newaxis]  # 提取H值
        bpkpk = torch.tensor(b_.max() - b_.min()).reshape(1, 1)
        # bh__ = np.concatenate((b_, h_), axis=1)
        # bh_ = np.stack((b_, h_)).reshape((2, 1024))
        # img = torch.tensor(img_)
        freq = torch.tensor(freq_)
        # freq_trans = torch.tensor(freq_log10)
        t = torch.tensor(t_)
        b = torch.tensor(b_)
        h = torch.tensor(h_)

        norm_b = (b_ - self.norm_dict["mean_B"]) / self.norm_dict["std_B"]
        norm_h = (h_ - self.norm_dict["mean_H"]) / self.norm_dict["std_H"]
        norm_t = (t_ - self.norm_dict["mean_T"]) / self.norm_dict["std_T"]
        norm_freq_log10 = (freq_log10 - self.norm_dict["mean_freq_log10"]) / self.norm_dict["std_freq_log10"]

        norm_b_nt = torch.tensor(norm_b)
        norm_h_nt = torch.tensor(norm_h)
        norm_t_nt = torch.tensor(norm_t).reshape(1, )
        norm_freq_log10_nt = torch.tensor(norm_freq_log10).reshape(1, )

        # 如果要在dataloader阶段就构建noise序列的话那么应该加上的大小为一个patch的大小16/32
        #
        # head = 0.1 * torch.ones(h.size()[0], h.size()[1])
        # noise_h = head + (torch.rand(h.size()) - 0.5) * 0.1  # 随机噪声序列H
        # 验证集的h应该是一个随机序列，其实可以在验证时随机生成,写在predict方法里面
        head = 0.1 * torch.ones(16, norm_h_nt.size()[1])
        h_head = torch.cat((head, norm_h_nt), dim=0)
        h_noize = h_head + (torch.rand(h_head.size()) - 0.5) * 0.1

        ctarget = np.array(self.target_trans(self.labels[index])).reshape(1)
        ctarget_log10 = np.log10(ctarget)
        dtarget = np.array(self.target_trans(self.dlabels[index])).reshape(1)
        # return b, h, freq, t, ctarget, dtarget, bh, bh_nt, ctarget_trans, bpkpk
        # return b, h, freq, t, ctarget, dtarget, bh_nt, ctarget_trans, bpkpk, freq_trans, norm_bh_nt, norm_t_nt, norm_freq_log10_nt
        return b, h, freq, t, ctarget, dtarget, norm_b_nt, norm_h_nt, norm_t_nt, norm_freq_log10_nt, bpkpk, h_head, h_noize

    def __len__(self):
        return len(self.indices)

    def test(self, index):
        index = self.indices[index]
        # img = self.input_trans(self.loader(self.x[index]))
        img_ = self.input_trans(self.loader(self.x[index]))

        return img_


if __name__ == '__main__':
    norm = init_norm_dict("D:\\PycharmProjects\\dg-magnet\\variables.npz")
    testData = SeriesDatasetLSTMNormb(dataset='test_dataset', task='test_task',
                                      root_dir='D:\\PycharmProjects\\dg-magnet\\', domain_name='N27', norm_dict=norm)

    # index = testData.indices[0]
    # img_ = testData.input_trans(testData.loader(testData.x[index]))
    #
    # freq_ = np.array(img_[0]).reshape(1)
    # freq_log10 = np.log10(freq_)
    # t_ = np.array(img_[1]).reshape(1)
    # b_ = img_[2:1026][:, np.newaxis]  # 提取B值
    # bpkpk = b_.max() - b_.min()
    # h_ = img_[1026:2050][:, np.newaxis]  # 提取H值
    #
    # t1 = norm["mean_B"]
    # t2 = norm["std_B"]
    #
    # norm_b = (b_ - t1) / t2
    #
    # t3 = norm["mean_H"]
    # t4 = norm["std_H"]
    #
    # norm_h = (h_ - t3) / t4
    #
    # t5 = norm["mean_T"]
    # t6 = norm["std_T"]
    #
    # norm_t = (t_ - t5) / t6
    #
    # t7 = norm["mean_freq_log10"]
    # t8 = norm["std_freq_log10"]
    #
    # norm_freq_log10 = (freq_log10 - t7) / t8

    # 然后他们用的F的norm是先log10然后再norm，应该norm字典里面已经就是log10后的统计值了

    a = testData.__getitem__(0)
    data_len = testData.__len__()

    # b = a[0].reshape(-1, )
    # h = a[1].reshape(-1, )
    # bh = np.trapz(b, h, axis=1)
