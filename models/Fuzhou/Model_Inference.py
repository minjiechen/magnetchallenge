# -*- encoding: utf-8 -*-
'''
Filename         :Model_Inference.py
Description      :Testing the model on Testing data.
Author           :Fuzhou University
'''

import os
from typing import Any
import numpy as np
import pandas as pd

import lightning.pytorch as pl
import torchinfo

from MagNet_Data import MagNetDataModule
from MagNet_Model import Transformer, Lit_model
    

Material = 'E'

#%% Load Dataset
DATA_ROOT = r'/root/autodl-tmp/data/final_test/' # The data directory in the test PC

def load_dataset(in_file1=DATA_ROOT+rf'Testing/Material {Material}/B_Field.csv',
                 in_file2=DATA_ROOT+rf'Testing/Material {Material}/Frequency.csv',
                 in_file3=DATA_ROOT+rf'Testing/Material {Material}/Temperature.csv',):
    data_B = pd.read_csv(in_file1, header=None)
    data_F = pd.read_csv(in_file2, header=None)
    data_T = pd.read_csv(in_file3, header=None)

    return data_B, data_F, data_T

#%%

def core_loss(data_B, data_F, data_T):

    # Create Pytorch Lightning Dataset
    # Note: If the test PC dont have enough GPU memory, set a smaller [batch_size].
    #       However, changing the [batch_size] will make a small difference in performance.
    print('------------/ Start load dataset... /------------')
    dm = MagNetDataModule(data_B, data_F, data_T, batch_size=128,
                          norm_info_path=rf'./Model/norm_info_{Material}.json')
    dm.prepare_data()
    dm.setup('inference')
    print('------------/ Successfully load dataset! /------------')

    # Prepare Model
    print('------------/ Start prepare model...  /------------')
    net = Transformer()
    model = Lit_model(net, normF=dm.normF, normP=dm.normP)
    torchinfo.summary(model) # print Mem. for each layer
    
    trainer = pl.Trainer(
        accelerator="gpu",
        benchmark=False,
        deterministic=True,
        precision='16-mixed',
        logger=False
    )
    print('------------/ Successfully load model! /------------')

    # Inference
    print('------------/ Start inference... /------------')
    trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path=rf'./Model/Material {Material}.ckpt')

    # Save Results
    os.makedirs('./result', exist_ok=True)
    data_P = model.results
    with open(rf'./result/Volumetric_Loss_{Material}.csv', "w") as f:
        np.savetxt(f, data_P)
        f.close()

    print('------------/ Model validation is finished! /------------')


#%%
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    pl.seed_everything(666) # reproducibility

    data_B, data_F, data_T = load_dataset()

    core_loss(data_B, data_F, data_T)
    
if __name__ == '__main__':
    main()
