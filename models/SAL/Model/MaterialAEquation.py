"""########################################################################
#
#             Title : Function using a NN to compute the magnetic core loss
#
###########################################################################
#
#            Author : Jacob Reynvann, Martin Stoiber
#
########################################################################"""
import numpy as np
# local imports
from StoiberReynvannEquation import StoiberReynvannEquation, load_data

LIBRARY_PATH = 'D:/Projekte/MagNet23 Challenge/MagNet 2023 Challenge Testing Data (Public)/Testing/'

def MaterialAEquation(freq: float, temp: float, B_wf: list) -> float:
    return StoiberReynvannEquation(freq=freq, temp=temp, B_wf=B_wf, materialId=10)

def MaterialA2CSV(folder_path: str) -> float:
    freq, temp, B_wf = load_data(meas_path=folder_path)
    VLoss = np.zeros_like(freq)
    for idx, (f,t,B) in enumerate(zip(freq,temp,B_wf)):
        VLoss[idx] = MaterialAEquation(freq=f, temp=t, B_wf=B)
    VLoss.reshape((1,-1))
    np.savetxt(f'./Material A.csv', VLoss)

if __name__ == '__main__':
    MaterialA2CSV(folder_path=LIBRARY_PATH+'Material A/')