# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader

import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset
from datautil.seriesdata.seriesdataload import SeriesDataset
from datautil.seriesdata.seriesdataload import SeriesDatasetLSTM, SeriesDatasetValid
from datautil.seriesdata.seriesdataload_norm import SeriesDatasetLSTMNorm, SeriesDatasetValidNorm
from datautil.seriesdata.seriesdataload_norm import SeriesDatasetLSTMNormb, SeriesDatasetValidNormb
from datautil.seriesdata.seriesdataload_simple import SeriesDatasetFile, SeriesDatasetFileFinal

from datautil.mydataloader import InfiniteDataLoader
from utils.util import init_norm_dict


def get_img_dataloader(args):
    rate = 0.2
    trdatalist, tedatalist = [], []

    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        if i in args.test_envs:
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_test(args.dataset),
                                           test_envs=args.test_envs))
        else:
            tmpdatay = ImageDataset(args.dataset, args.task, args.data_dir,
                                    names[i], i, transform=imgutil.image_train(args.dataset),
                                    test_envs=args.test_envs).labels
            l = len(tmpdatay)
            if args.split_style == 'strat':
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size=rate, train_size=1 - rate, random_state=args.seed)
                stsplit.get_n_splits(lslist, tmpdatay)
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                indexall = np.arange(l)
                np.random.seed(args.seed)
                np.random.shuffle(indexall)
                ted = int(l * rate)
                indextr, indexte = indexall[:-ted], indexall[-ted:]

            trdatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_train(args.dataset), indices=indextr,
                                           test_envs=args.test_envs))
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_test(args.dataset), indices=indexte,
                                           test_envs=args.test_envs))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in trdatalist + tedatalist]

    return train_loaders, eval_loaders


def get_series_dataloader(args):
    rate = 0.2
    trdatalist, tedatalist = [], []

    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        if i in args.test_envs:
            # tedatalist.append(SeriesDataset(args.dataset, args.task, args.data_dir,
            #                                names[i], i, transform=imgutil.image_test(args.dataset), test_envs=args.test_envs))
            tedatalist.append(SeriesDataset(args.dataset, args.task, args.data_dir,
                                            names[i], i, test_envs=args.test_envs))
        else:
            # tmpdatay = SeriesDataset(args.dataset, args.task, args.data_dir,
            #                         names[i], i, transform=imgutil.image_train(args.dataset), test_envs=args.test_envs).labels
            tmpdatay = SeriesDataset(args.dataset, args.task, args.data_dir,
                                     names[i], i, test_envs=args.test_envs).labels
            l = len(tmpdatay)
            # if args.split_style == 'strat':   # 这段应该要改
            #     lslist = np.arange(l)
            #     stsplit = ms.StratifiedShuffleSplit(
            #         2, test_size=rate, train_size=1-rate, random_state=args.seed)
            #     stsplit.get_n_splits(lslist, tmpdatay)
            #     indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            # else:
            # 暂时不符合StratifiedShuffleSplit方法格式,先弃用使用普通split方法
            indexall = np.arange(l)
            np.random.seed(args.seed)
            np.random.shuffle(indexall)
            ted = int(l * rate)
            indextr, indexte = indexall[:-ted], indexall[-ted:]

            # trdatalist.append(SeriesDataset(args.dataset, args.task, args.data_dir,
            #                                names[i], i, transform=imgutil.image_train(args.dataset), indices=indextr, test_envs=args.test_envs))
            # tedatalist.append(SeriesDataset(args.dataset, args.task, args.data_dir,
            #                                names[i], i, transform=imgutil.image_test(args.dataset), indices=indexte, test_envs=args.test_envs))
            trdatalist.append(SeriesDataset(args.dataset, args.task, args.data_dir,
                                            names[i], i, indices=indextr, test_envs=args.test_envs))
            tedatalist.append(SeriesDataset(args.dataset, args.task, args.data_dir,
                                            names[i], i, indices=indexte, test_envs=args.test_envs))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in trdatalist + tedatalist]

    return train_loaders, eval_loaders


def get_LSTM_dataloader(args):
    rate = 0.2
    trdatalist, tedatalist = [], []

    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        if i in args.test_envs:
            # tedatalist.append(SeriesDataset(args.dataset, args.task, args.data_dir,
            #                                names[i], i, transform=imgutil.image_test(args.dataset), test_envs=args.test_envs))
            tedatalist.append(SeriesDatasetLSTM(args.dataset, args.task, args.data_dir,
                                                names[i], i, test_envs=args.test_envs))
        else:
            # tmpdatay = SeriesDataset(args.dataset, args.task, args.data_dir,
            #                         names[i], i, transform=imgutil.image_train(args.dataset), test_envs=args.test_envs).labels
            tmpdatay = SeriesDatasetLSTM(args.dataset, args.task, args.data_dir,
                                         names[i], i, test_envs=args.test_envs).labels
            l = len(tmpdatay)
            # if args.split_style == 'strat':   # 这段应该要改
            #     lslist = np.arange(l)
            #     stsplit = ms.StratifiedShuffleSplit(
            #         2, test_size=rate, train_size=1-rate, random_state=args.seed)
            #     stsplit.get_n_splits(lslist, tmpdatay)
            #     indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            # else:
            # 暂时不符合StratifiedShuffleSplit方法格式,先弃用使用普通split方法
            indexall = np.arange(l)
            np.random.seed(args.seed)
            np.random.shuffle(indexall)
            ted = int(l * rate)
            indextr, indexte = indexall[:-ted], indexall[-ted:]

            # trdatalist.append(SeriesDataset(args.dataset, args.task, args.data_dir,
            #                                names[i], i, transform=imgutil.image_train(args.dataset), indices=indextr, test_envs=args.test_envs))
            # tedatalist.append(SeriesDataset(args.dataset, args.task, args.data_dir,
            #                                names[i], i, transform=imgutil.image_test(args.dataset), indices=indexte, test_envs=args.test_envs))
            trdatalist.append(SeriesDatasetLSTM(args.dataset, args.task, args.data_dir,
                                                names[i], i, indices=indextr, test_envs=args.test_envs))
            tedatalist.append(SeriesDatasetLSTM(args.dataset, args.task, args.data_dir,
                                                names[i], i, indices=indexte, test_envs=args.test_envs))

    # train_loaders = [DataLoader(
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in trdatalist + tedatalist]

    return train_loaders, eval_loaders


def get_train_dataloader(args):
    trdatalist = []
    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        trdatalist.append(SeriesDatasetLSTM(args.dataset, args.task, args.data_dir,
                                            names[i], i, test_envs=args.test_envs))
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]
    return train_loaders


def get_eval_dataloader(args):
    tedatalist = []
    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        tedatalist.append(SeriesDatasetValid(args.dataset, args.task, args.data_dir,
                                             names[i], i, test_envs=args.test_envs))
    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in tedatalist]
    return eval_loaders


def get_train_dataloader_norm(args, norm_dict):
    trdatalist = []
    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        trdatalist.append(SeriesDatasetLSTMNorm(args.dataset, args.task, args.data_dir,
                                                names[i], i, test_envs=args.test_envs, norm_dict=norm_dict))
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]
    return train_loaders


def get_eval_dataloader_norm(args, norm_dict):
    tedatalist = []
    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        tedatalist.append(SeriesDatasetValidNorm(args.dataset, args.task, args.data_dir,
                                                 names[i], i, test_envs=args.test_envs, norm_dict=norm_dict))
    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in tedatalist]
    return eval_loaders


def get_train_dataloader_normb(args, norm_dict):
    trdatalist = []
    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        tmpdatay = SeriesDatasetLSTMNormb(args.dataset, args.task, args.data_dir,
                                          names[i], i, test_envs=args.test_envs).labels
        l = len(tmpdatay)
        indexall = np.arange(l)
        np.random.seed(args.seed)
        np.random.shuffle(indexall)

        trdatalist.append(SeriesDatasetLSTMNormb(args.dataset, args.task, args.data_dir,
                                                 names[i], i, indices=indexall, test_envs=args.test_envs,
                                                 norm_dict=norm_dict))
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]
    return train_loaders


def get_eval_dataloader_normb(args, norm_dict):
    tedatalist = []
    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        tedatalist.append(SeriesDatasetValidNormb(args.dataset, args.task, args.data_dir,
                                                  names[i], i, test_envs=args.test_envs, norm_dict=norm_dict))
    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in tedatalist]
    return eval_loaders


def get_train_dataloader_simple(args, norm_dict, mode="train"):
    trdatalist = []
    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        tmpdatay = SeriesDatasetFile(args.dataset, args.task, args.data_dir,
                                     names[i], i, test_envs=args.test_envs, mode=mode, norm_dict=norm_dict).labels
        l = len(tmpdatay)
        indexall = np.arange(l)
        np.random.seed(args.seed)
        np.random.shuffle(indexall)

        trdatalist.append(SeriesDatasetFile(args.dataset, args.task, args.data_dir,
                                            names[i], i, indices=indexall, test_envs=args.test_envs,
                                            mode=mode, norm_dict=norm_dict))
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]
    return train_loaders


def get_eval_dataloader_simple(args, norm_dict, mode="eval"):
    tedatalist = []
    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        tedatalist.append(SeriesDatasetFile(args.dataset, args.task, args.eval_dir,
                                            names[i], i, test_envs=args.test_envs, norm_dict=norm_dict))
    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in tedatalist]
    return eval_loaders


def get_eval_dataloader_simple_final(args, norm_dict):
    tedatalist = []
    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        tedatalist.append(SeriesDatasetFileFinal(args.dataset, args.task, args.eval_dir,
                                                 names[i], i, test_envs=args.test_envs, norm_dict=norm_dict))
    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in tedatalist]
    return eval_loaders


def get_final_dataloader(args, norm_dict):
    rate = 0.1
    trdatalist, tedatalist = [], []

    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        tmpdatay = SeriesDatasetFile(args.dataset, args.task, args.data_dir,
                                     names[i], i, test_envs=args.test_envs, mode="final", norm_dict=norm_dict).labels
        l = len(tmpdatay)

        indexall = np.arange(l)
        np.random.seed(args.seed)
        np.random.shuffle(indexall)
        ted = int(l * rate)
        indextr, indexte = indexall[:-ted], indexall[-ted:]

        trdatalist.append(SeriesDatasetFile(args.dataset, args.task, args.data_dir,
                                            names[i], i, indices=indextr, test_envs=args.test_envs, mode="final",
                                            norm_dict=norm_dict))
        tedatalist.append(SeriesDatasetFile(args.dataset, args.task, args.data_dir,
                                            names[i], i, indices=indexte, test_envs=args.test_envs, mode="final",
                                            norm_dict=norm_dict))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in tedatalist]

    return train_loaders, eval_loaders


if __name__ == '__main__':
    norm = init_norm_dict("D:\\PycharmProjects\\dg-magnet\\variables.npz")
    # testData = SeriesDatasetSimple(dataset='test_dataset', task='test_task',
    #                                root_dir='D:\\PycharmProjects\\dg-magnet\\', domain_name='N27', norm_dict=norm)
    trdatalist = []
    names = ["3C90"]
    domain_num = len(names)
    for i in range(len(names)):
        # generate for split
        tmpdatay = SeriesDatasetFile(dataset="MAGNET_3C90", task="img_dg",
                                     root_dir='D:\\PycharmProjects\\dg-magnet\\MagNet 2023 Database\\',
                                     domain_name=names[i], domain_label=i, test_envs=[], norm_dict=norm
                                     , mode="train").labels
        l = len(tmpdatay)
        indexall = np.arange(l)
        np.random.seed(0)
        np.random.shuffle(indexall)

        trdatalist.append(SeriesDatasetFile(dataset="MAGNET_3C90", task="img_dg",
                                            root_dir='D:\\PycharmProjects\\dg-magnet\\MagNet 2023 Database\\',
                                            domain_name=names[i], domain_label=i, indices=indexall,
                                            test_envs=[],
                                            norm_dict=norm,
                                            mode="train"))
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=32,
        num_workers=4)
        for env in trdatalist]

    test_loader = [DataLoader(
        dataset=env,
        batch_size=32,
        num_workers=4,
        drop_last=False,
        shuffle=False)
        for env in trdatalist]

    loader_unit = test_loader[0]

    data = next(iter(loader_unit))
