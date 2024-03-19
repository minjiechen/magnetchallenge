"""########################################################################
#
#             Title : Function using a NN to compute the magnetic core loss
#
###########################################################################
#
#            Author : Jacob Reynvann, Martin Stoiber
#             Notes : The function which computes the core loss for all 
#                     materials is called `StoiberReynvannEquation`.
#       MaterialIds : Material | Id
#                         A    | 10
#                         B    | 11
#                         C    | 12
#                         D    | 13
#                         E    | 14
#
########################################################################"""
import os
import numpy as np
import pandas as pd
import torch
from torch import nn

LIBRARY_PATH = 'D:/Projekte/MagNet23 Challenge/MagNet 2023 Challenge Testing Data (Public)/Testing/'
MODEL_NAME = 'weights_of_Epoch_new_3136.pt'
DTYPE = np.float32

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(516, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
        )

    @classmethod
    def load_model(cls, filename, device='cpu'):
        model = cls().to(device)
        model.load_state_dict(torch.load(filename, map_location=device))
        return model
    
    def forward(self, x):
        return self.linear_relu_stack(x)

    # Predict using the loaded model
    def predict(self, data):
        # self.eval()
        with torch.no_grad():
            output = self(torch.tensor(data, dtype=torch.float32))
        return output
        

def StoiberReynvannEquation(freq: float, temp: float, B_wf: list, materialId: int) -> float:
    #region prepare data
    B_wf = np.interp(x=np.arange(256), xp=np.arange(len(B_wf)), fp=B_wf)
    freq_fft = np.fft.fftfreq(n=len(B_wf), d=len(B_wf)/freq).astype(DTYPE, copy=False) + freq
    B_wf_fft = np.abs(np.fft.fft(B_wf)).astype(DTYPE, copy=False)
    n_datapoints = len(freq_fft)
    data = {}
    data['temp'] = temp
    data['materialId'] = materialId
    data['fftval_scaler'] = np.max(B_wf_fft)
    data['fftsup_scaler'] = np.max(freq_fft)
    for i in range(n_datapoints):
        data[f'fftval_{i}'] = B_wf_fft[i] / data['fftval_scaler']
        data[f'fftsup_{i}'] = freq_fft[i] / data['fftsup_scaler']
    ds = pd.Series(data, dtype=DTYPE)
    #endregion
    #region scale data
    df_min = pd.read_pickle('./scaling_min.pkl')
    VLoss_min = df_min.pop('VLoss')
    df_max = pd.read_pickle('./scaling_max.pkl')
    VLoss_max = df_max.pop('VLoss')
    normalized_ds = (ds-df_min) / (df_max-df_min)
    for i in range(n_datapoints):
        normalized_ds[f'fftval_{i}'] = ds[f'fftval_{i}']
        normalized_ds[f'fftsup_{i}'] = ds[f'fftsup_{i}']
    #endregion
    #region predict data
    model = NeuralNetwork.load_model(filename=MODEL_NAME)
    VLoss_scaled = model.predict(data=normalized_ds.to_numpy()).numpy()[0]
    VLoss = VLoss_scaled*(VLoss_max-VLoss_min)+VLoss_min
    #endregion

    return VLoss

def load_data(meas_path:str) -> tuple:
    def read_csv(fname:str) -> np.array:
        file_path = os.path.join(meas_path, fname)
        return pd.read_csv(file_path, header=None).to_numpy()

    freq = read_csv('Frequency.csv').flatten()
    B_wf = read_csv('B_Field.csv')
    temp = read_csv('Temperature.csv').flatten()

    return freq, temp, B_wf

if __name__ == '__main__':
    import logging
    from tqdm import tqdm
    for materialId, material in enumerate(os.listdir(LIBRARY_PATH)):
        logging.info(f"inspect material {material}")
        if os.path.isdir(LIBRARY_PATH+material):
            freq, temp, B_wf = load_data(LIBRARY_PATH+material)
            VLoss = np.zeros_like(freq)
            for idx, (f,t,B) in tqdm(list(enumerate(zip(freq,temp,B_wf)))):
                VLoss[idx] = StoiberReynvannEquation(f,t,B,10+materialId)

            VLoss.reshape((1,-1))
            np.savetxt(f'./{material}.csv', VLoss)