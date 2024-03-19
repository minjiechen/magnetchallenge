import math
import numpy as np



#%% count number of parameters of neural network
def downsample(signal, factor):
    length = math.floor(len(signal)/factor)
    signal_new = [signal[round(n*factor)] for n in range(length)]
    return signal_new

#%% downsample 
def SeqDownsample(Seq, f, f_min_):
    # downsampling
    for i in range(len(Seq)):
        input_signal = Seq[i]
        rate = float(f[i]/f_min_)
        downsampled_signal = downsample(input_signal, rate)
        downsampled_signal = math.ceil(rate)*downsampled_signal
        Seq[i] = downsampled_signal[:1024]
    return Seq

#%% Add Noise on waveforms
def AddNoise(Seq):
    noisy_weights = []
    for i in range(len(Seq)):
        std = np.std(Seq[i])
        noise = np.random.normal(loc=0.0, scale=std, size=Seq[i].shape)
        noisy_weights.append(Seq[i]*1000 + noise*5)
    Seq = np.array(noisy_weights)
    return Seq

#%% make random phase shift on waveforms
def random_phase(Seq):
    for i in range(np.shape(Seq)[0]):
        shift = np.random.randint(0,1024)
        row = Seq[i]
        row_new = np.append(row[1024-shift:],row[:1024-shift])
        Seq[i] = row_new
    return Seq

