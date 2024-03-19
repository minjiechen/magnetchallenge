# coding=utf-8
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



def SeriesFolder(folder_path):
    List_tuple = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            _ = np.load(file_path)["data"]
            __ = _[-1]
            List_tuple.append((file_path, __))
            # print(file_path)
    return List_tuple


class SeriesDataset(Dataset):
    def __init__(self, dataset, task, root_dir, domain_name, domain_label=-1, labels=None, transform=None,
                 target_transform=None, indices=None, test_envs=[], mode='Default'):
        # self.datas = SeriesFolder("D:\PycharmProjects\\bh_loss\domain_data_npz\\3C90")
        print(root_dir + " domain_data_npz_\\" + domain_name)
        self.datas = SeriesFolder(root_dir + "domain_data_npz_\\" + domain_name)
        # self.imgs = ImageFolder(root_dir+domain_name).imgs
        # self.imgs = ImageFolder(root_dir+domain_name).imgs
        self.domain_num = 0
        self.task = task
        self.dataset = dataset
        imgs = [item[0] for item in self.datas]
        labels = [item[1] for item in self.datas]
        self.labels = np.array(labels)
        # self.labels = np.array(labels) / 100
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
        img__ = np.pad(img_, (0, 3 * 32 * 32 - img_.size), mode='constant').reshape(3, 32, 32)
        img = torch.tensor(img__)
        ctarget = self.target_trans(self.labels[index])
        dtarget = self.target_trans(self.dlabels[index])
        return img, ctarget, dtarget

    def __len__(self):
        return len(self.indices)


class SeriesDatasetLSTM(Dataset):
    def __init__(self, dataset, task, root_dir, domain_name, domain_label=-1, labels=None, transform=None,
                 target_transform=None, indices=None, test_envs=[],mode='Default'):
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
        # self.labels = np.array(labels) / 100
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
        bpkpk = b_.max() - b_.min()
        h_ = img_[1026:2050][:, np.newaxis]  # 提取H值
        bh__ = np.concatenate((b_, h_), axis=1)
        # bh_ = np.stack((b_, h_)).reshape((2, 1024))
        # img = torch.tensor(img_)
        freq = torch.tensor(freq_)
        freq_trans = torch.tensor(freq_log10)
        t = torch.tensor(t_)
        b = torch.tensor(b_)
        h = torch.tensor(h_)
        # bh = torch.tensor(bh_)
        bh_nt = torch.tensor(bh__)

        ctarget = np.array(self.target_trans(self.labels[index])).reshape(1)
        ctarget_trans = np.log10(ctarget)
        dtarget = np.array(self.target_trans(self.dlabels[index])).reshape(1)
        # return b, h, freq, t, ctarget, dtarget, bh, bh_nt, ctarget_trans, bpkpk
        return b, h, freq, t, ctarget, dtarget, bh_nt, ctarget_trans, bpkpk, freq_trans

    def __len__(self):
        return len(self.indices)

    def test(self, index):
        index = self.indices[index]
        # img = self.input_trans(self.loader(self.x[index]))
        img_ = self.input_trans(self.loader(self.x[index]))

        return img_


class SeriesDatasetValid(Dataset):
    def __init__(self, dataset, task, root_dir, domain_name, domain_label=-1, labels=None, transform=None,
                 target_transform=None, indices=None, test_envs=[], mode='Default'):
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
        # self.labels = np.array(labels) / 100
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
        bpkpk = b_.max() - b_.min()
        h_ = img_[1026:2050][:, np.newaxis]  # 提取H值
        bh__ = np.concatenate((b_, h_), axis=1)
        # bh_ = np.stack((b_, h_)).reshape((2, 1024))
        # img = torch.tensor(img_)
        freq = torch.tensor(freq_)
        freq_trans = torch.tensor(freq_log10)
        t = torch.tensor(t_)
        b = torch.tensor(b_)
        h = torch.tensor(h_)
        # bh = torch.tensor(bh_)
        bh_nt = torch.tensor(bh__)

        ctarget = np.array(self.target_trans(self.labels[index])).reshape(1)
        ctarget_trans = np.log10(ctarget)
        dtarget = np.array(self.target_trans(self.dlabels[index])).reshape(1)
        # return b, h, freq, t, ctarget, dtarget, bh, bh_nt, ctarget_trans, bpkpk
        return b, h, freq, t, ctarget, dtarget, bh_nt, ctarget_trans, bpkpk, freq_trans

    def __len__(self):
        return len(self.indices)

    def test(self, index):
        index = self.indices[index]
        # img = self.input_trans(self.loader(self.x[index]))
        img_ = self.input_trans(self.loader(self.x[index]))

        return img_


if __name__ == '__main__':
    testData = SeriesDatasetLSTM(dataset='test_dataset', task='test_task',
                                 root_dir='D:\\PycharmProjects\\dg-magnet\\', domain_name='N27')

    a = testData.__getitem__(0)
    data_len = testData.__len__()
