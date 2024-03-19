import numpy as np
from scipy.signal import find_peaks
import torch
import scipy


def data_type_selection(in_B, threshold = 0.000203):
    in_B_type = []
    for i in range(in_B.size()[0]):
        # Calculate the derivative of the B signal
        x = in_B[i].numpy().squeeze()
        x = (x - np.min(x))/(np.max(x) - np.min(x))
        in_B_first_derivative = np.diff(x)
        in_B_second_derivative = np.abs(np.diff(in_B_first_derivative))
        spike_threshold = threshold # Adjust as needed
        distance_threshold = 10 # Adjust as needed
        spikes,_ = find_peaks(in_B_second_derivative,height = spike_threshold, distance = distance_threshold)
        
        index = []
        for i in range(len(spikes)-1):
            if abs(in_B_second_derivative[spikes[i]]) < abs(in_B_second_derivative[spikes[i+1]])/4:
                index.append(i)
        spikes = np.delete(spikes, index)

        # Check how many spikes on its second derivative to determine its signal type
        # 1 - Sinusoidal; 2 - Triangular; 3 - Trapezpidal
        if len(spikes) >= 3:
            in_B_type.append(3)
        elif len(spikes) >= 1:
            in_B_type.append(2)
        else:
            in_B_type.append(1)
    in_B_type = torch.FloatTensor(in_B_type).float().view(-1, 1)
    return in_B_type