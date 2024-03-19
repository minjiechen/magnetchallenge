import linear_std
import Maglib
import numpy as np

import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split


def dataTransform(raw_data, newStep, savePath,std_file_path='',std_loss_power=1.0):

    # de-sample data
    # b wave linear interpolation
    b_buff=np.zeros([raw_data.b.shape[0],newStep])

    for i in range(raw_data.b.shape[0]):
        x= np.linspace(0, newStep, raw_data.b.shape[1], endpoint=True)
        y= raw_data.b[i]

        k = newStep/raw_data.b.shape[1]
        b = np.interp(np.arange(0, newStep), x, y)

        b_buff[i]=b

    plt.plot(np.linspace(0, newStep, raw_data.b.shape[1], endpoint=True),
            raw_data.b[0],
            '.',
            label='raw data',
            color='red'
            )

    raw_data.b=b_buff

    plt.plot(np.linspace(0, newStep, raw_data.b.shape[1], endpoint=True),
            raw_data.b[0],
            'x',
            label='desampled data')

    plt.legend()
    plt.show()

    # standardize data
    if(std_file_path!=''):
        std_b=linear_std.linear_std()
        std_freq=linear_std.linear_std()
        std_temp=linear_std.linear_std()
        std_loss=linear_std.linear_std()

        std_b.load(std_file_path+r"\std_b.stdd")
        std_freq.load(std_file_path+r"\std_freq.stdd")
        std_temp.load(std_file_path+r"\std_temp.stdd")
        std_loss.load(std_file_path+r"\std_loss.stdd")
    else:
        std_b = linear_std.get_std_range(raw_data.b.min(), raw_data.b.max(), 0, 1)
        std_freq = linear_std.get_std_range(raw_data.freq.min(), raw_data.freq.max(), 0, 1)
        std_temp = linear_std.get_std_range(raw_data.temp.min(), raw_data.temp.max(), 0, 1)
        std_loss = linear_std.get_std_range(raw_data.loss.min(), raw_data.loss.max(), 0, 1)
    
    # add loss power to get better transform learning
    std_loss.k*=std_loss_power


    std_loss.b = 0  # make loss linear to get better relative loss
    std_b.b = 0  # make waveform linear to get better relative loss

    raw_data.freq = std_freq.std(raw_data.freq)
    raw_data.b = std_b.std(raw_data.b)
    raw_data.temp = std_temp.std(raw_data.temp)
    raw_data.loss = std_loss.std(raw_data.loss)
    raw_data.h=np.array(0.0)

    # let data type be float32
    raw_data.freq=raw_data.freq.astype(np.float32)
    raw_data.b=raw_data.b.astype(np.float32)
    raw_data.temp=raw_data.temp.astype(np.float32)
    raw_data.loss=raw_data.loss.astype(np.float32)
    raw_data.h=raw_data.h.astype(np.float32)


    raw_data.save2mat(savePath+r"\data_processed.mat")

    std_b.save(savePath+r"\std_b.stdd")
    std_freq.save(savePath+r"\std_freq.stdd")
    std_temp.save(savePath+r"\std_temp.stdd")
    std_loss.save(savePath+r"\std_loss.stdd")

    print("Data transform done")
    return raw_data


def dataSplit(raw_data, savePath, indice=[0.7, 0.2, 0.1]):

    generator=torch.Generator().manual_seed(0)

    allData=np.zeros([raw_data.b.shape[0],raw_data.b.shape[1]+3])

    # print(raw_data.freq.shape)

    allData[:,0:raw_data.b.shape[1]]=raw_data.b
    allData[:,raw_data.b.shape[1]]=raw_data.temp[:,0]
    allData[:,raw_data.b.shape[1]+1]=raw_data.loss[:,0]
    allData[:, raw_data.b.shape[1] + 2] = raw_data.freq[:, 0]


    train_set, valid_set,test_set = random_split(
        dataset=allData,
        lengths=indice,
        generator=generator)

    train_set=np.array(train_set,dtype=np.float32)
    valid_set=np.array(valid_set,dtype=np.float32)
    test_set = np.array(test_set, dtype=np.float32)


    stepLen=raw_data.b.shape[1]

    train_dataset = Maglib.MagLoader()
    train_dataset.b = train_set[:, 0:stepLen]
    train_dataset.temp = train_set[:, stepLen:stepLen + 1]
    train_dataset.loss = train_set[:, stepLen + 1:stepLen + 2]
    train_dataset.freq = train_set[:, stepLen + 2:stepLen + 3]
    train_dataset.h = np.array(0.0)

    train_dataset.save2mat(savePath+r"\train.mat")


    valid_dataset = Maglib.MagLoader()
    valid_dataset.b = valid_set[:, 0:stepLen]
    valid_dataset.temp = valid_set[:, stepLen:stepLen + 1]
    valid_dataset.loss = valid_set[:, stepLen + 1:stepLen + 2]
    valid_dataset.freq = valid_set[:, stepLen + 2:stepLen + 3]
    valid_dataset.h = np.array(0.0)

    valid_dataset.save2mat(savePath+r"\valid.mat")

    test_dataset = Maglib.MagLoader()
    test_dataset.b = test_set[:, 0:stepLen]
    test_dataset.temp = test_set[:, stepLen:stepLen + 1]
    test_dataset.loss = test_set[:, stepLen + 1:stepLen + 2]
    test_dataset.freq = test_set[:, stepLen + 2:stepLen + 3]
    test_dataset.h = np.array(0.0)

    test_dataset.save2mat(savePath + r"\test.mat")

    print("DataSplit done")


if __name__ == '__main__':

    # init parameters
    raw_data = Maglib.MagLoader(r"data\raw\78_cycle.mat")
    newStep=128
    savePath=r"data\std_78_cycle"


    raw_data=dataTransform(raw_data, newStep, savePath)
    dataSplit(raw_data, savePath)
    print("Done")