"""
Version: 1.5
    able to load from new version of MagNet raw csv file

Version: 1.4
    Add function to load from MagNet raw csv file

Version: 1.3
    Add empty load function

Version: 1.2
    change the way to import data of MagLoader
    Add version info in Maglib.py

Version: 1.1
    Function added: MagPlot2

Version: 1.0
    MagLoader finished construction
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


class MagLoader:

    b = 0
    h = 0
    temp = 0
    loss = 0
    freq = 0


    def __init__(self,material_path='',data_type='numpy',data_source='mat'):
        # Load .mat file

        if material_path!='':
            if data_source=='mat':
                data = sio.loadmat(material_path)

                # Convert to NumPy array

                self.b = np.array(data['b'])
                self.h = np.array(data['h'])
                self.temp = np.array(data['temp'])
                self.loss = np.array(data['loss'])
                self.freq = np.array(data['freq'])
            elif data_source=='csv':
                # self.b = np.loadtxt(material_path+r'\B_waveform[T].csv', delimiter=',').astype(np.float32)
                # self.h = np.loadtxt(material_path+r'\H_waveform[Am-1].csv', delimiter=',').astype(np.float32)
                # self.temp = (np.loadtxt(material_path+r'\Temperature[C].csv', delimiter=',') + 273.15).astype(np.float32)
                # self.freq = np.loadtxt(material_path+r'\Frequency[Hz].csv', delimiter=',').astype(np.float32)
                # self.loss = np.loadtxt(material_path+r'\Volumetric_losses[Wm-3].csv', delimiter=',').astype(np.float32)

                # self.b = np.loadtxt(material_path+r'\B_waveform.csv', delimiter=',').astype(np.float32)
                # self.h = np.loadtxt(material_path+r'\H_waveform.csv', delimiter=',').astype(np.float32)
                # self.temp = (np.loadtxt(material_path+r'\Temperature.csv', delimiter=',') + 273.15).astype(np.float32)
                # self.freq = np.loadtxt(material_path+r'\Frequency.csv', delimiter=',').astype(np.float32)
                # self.loss = np.loadtxt(material_path+r'\Volumetric_losses.csv', delimiter=',').astype(np.float32)

                self.b = np.loadtxt(material_path+r'\B_Field.csv', delimiter=',').astype(np.float32)
                self.h = np.loadtxt(material_path+r'\H_Field.csv', delimiter=',').astype(np.float32)
                self.temp = (np.loadtxt(material_path+r'\Temperature.csv', delimiter=',') + 273.15).astype(np.float32)
                self.freq = np.loadtxt(material_path+r'\Frequency.csv', delimiter=',').astype(np.float32)
                self.loss = np.loadtxt(material_path+r'\Volumetric_Loss.csv', delimiter=',').astype(np.float32)

                self.temp=self.temp[:,np.newaxis]
                self.freq=self.freq[:,np.newaxis]
                self.loss=self.loss[:,np.newaxis]
        


            if data_type == 'torch':
                import torch
                # Convert to PyTorch tensor
                self.b=torch.from_numpy(self.b)
                self.h=torch.from_numpy(self.h)
                self.temp=torch.from_numpy(self.temp)
                self.loss=torch.from_numpy(self.loss)
                self.freq=torch.from_numpy(self.freq)
            
        else:
            pass

        return

    def save2mat(self,save_path):
        # Save to .mat file
        sio.savemat(save_path, {'b': self.b, 'h': self.h, 'temp': self.temp, 'loss': self.loss, 'freq': self.freq})
        return

class MagPlot:

    MagData = 0

    def __init__(self, material_path):
        self.MagData = MagLoader(material_path)
        pass

    def plot(self,idx):
        B=self.MagData.b[idx]
        H=self.MagData.h[idx]
        freq=float(self.MagData.freq[idx])
        loss=float(self.MagData.loss[idx])
        temp=float(self.MagData.temp[idx])

        info = f"loss: {loss:.2e}W/m3\nTemp: {temp-273.15:.0f}deg\nFreq: {freq/1000:.2f} kHz"

        # Create a time vector
        t = np.linspace(0, 1/freq, len(B))


        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))

        # Plot B-H loop
        axs[0].plot(H, B)
        axs[0].set_xlabel('H (A/m)')
        axs[0].set_ylabel('B (T)')
        axs[0].set_title('B-H Loop \n'+info)
        axs[0].grid(True)

        # Plot B waveform in the time domain
        axs[1].plot(t, B)
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('B')
        axs[1].set_title('B(t)')
        axs[1].grid(True)

        # Plot H waveform in the time domain
        axs[2].plot(t, H)
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('H')
        axs[2].set_title('H(t)')
        axs[2].grid(True)

        # Adjust the layout
        plt.tight_layout()

        # Show the plot
        plt.show()

        pass

    def plot2(self, idx):
        B = self.MagData.b[idx]
        H = self.MagData.h[idx]
        freq = float(self.MagData.freq[idx])
        loss = float(self.MagData.loss[idx])
        temp = float(self.MagData.temp[idx])

        info = f"loss: {loss:.2e}W/m3\nTemp: {temp-273.15:.0f}deg\nFreq: {freq/1000:.2f} kHz"

        # Create a time vector
        t = np.linspace(0, 1 / freq, len(B))

        # Create subplots
        # fig, axs = plt.subplots(1, 2, figsize=(8, 12))
        fig, axs = plt.subplots(2, 1, figsize=(9, 6))


        # Plot B-H loop
        axs[0].plot(H, B)
        axs[0].set_xlabel('H (A/m)')
        axs[0].set_ylabel('B (T)')
        axs[0].set_title('B-H Loop \n' + info)
        axs[0].grid(True)

        # Plot B waveform in the time domain
        axs[1].plot(t, B, label='B(t)',color='blue')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('B(t)',color='blue')
        axs[1].set_title('B(t) & H(t)')
        axs[1].tick_params(axis='y',colors='blue')

        axs[1].grid(True)

        # Plot H waveform in the time domain
        axs2=axs[1].twinx()
        axs2.plot(t, H, label='H(t)',color='orange')
        axs2.set_xlabel('Time')
        axs2.set_ylabel('H(t)',color='orange')
        #axs2.set_title('H(t)')

        axs2.tick_params(axis='y',colors='orange')
        axs2.grid(True)

        # Adjust the layout
        plt.tight_layout()

        # Show the plot
        plt.show()

        pass



# Demo
if __name__ == '__main__':

    # MagLoader()
    a = MagLoader(
        r"D:\OneDrive - University of Bristol\张力中的体系\项目\MagNet\DataSet\raw\77_cycle.mat"
    )

    # MagPlot()
    b=MagPlot(r"D:\OneDrive - University of Bristol\张力中的体系\项目\MagNet\DataSet\raw\77_cycle.mat")
    b.plot(2000)

    #linear_std
    rangeTest = getRange(10, 15, 0, 100)
    rangeTest.b,rangeTest.k

    rangeTest.std(11),rangeTest.std(14)
