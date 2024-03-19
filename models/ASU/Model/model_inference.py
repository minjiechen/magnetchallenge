"""
Model inference with a template from Magnet Challenge github.

    + models are stored as .json files in v0.0r (versioned) folder
        + version subfolder might be used
    + Change Material variable to match a material in test set
    + Change version if multiple version of model are supported.
        (one version for submission)
    + to test accuracy, one would need to use B and H from
        training to get if the output model model

"""

import random
import numpy as np
import json
import os
import sys

# To allow calling the script from any directory.
filepath = os.path.realpath(__file__)
# Get the directory of the current file
scriptdir = os.path.dirname(filepath)
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')

Material = "Material D"
Version = "v0.0r"  # internal to team


nparams = 24  # Don't change
#  Utility functions


def rotate2D(y, degrees, fast=True):
    """shift by a given degree. along axis=1, (2D array, assumed).
    """
    points = int(degrees/360*y.shape[1])
    array = []

    part2 = y[:, :-points]
    part1 = y[:, -points:]
    if fast:
        return np.concatenate([part1, part2], axis=1)

    for i in range(points, len(y) + points):
        array.append(y[i - len(y)])
    return np.array(array)


def shift_data(data):
    """
    shift the data to start at the min value
    """
    new = []
    if data.ndim == 2:
        locMin = np.argmin(data, axis=1)

        for i in range(len(data)):
            locMini = np.argmin(data, axis=1)[i]
            degrees = 360-locMini/len(data[i, :])*360
            # print(degrees)
            if locMini == 0 or locMini == len(data):
                new.append(np.array([data[i]]))
            else:
                new.append(rotate2D(data[i].reshape(1, -1),
                           degrees).reshape(-1))
        return np.array(new)

    if data.ndim == 1:
        locMin = np.argmin(data)
        degrees = 360-locMin/len(data)*360
        print(degrees)
        new = rotate2D(np.array([data]), degrees)[0]
        return new


def OneHotEncoder(T=[25, 50, 70, 95]):
    """ simplified onehot encoder. Assumption is data only
        consists of the 4 temperatures
    """
    label = np.zeros((len(T), 4))
    T = T.reshape(-1)
    for i in range(len(T)):
        if T[i] <= 25:
            y = 0
        elif T[i] < 51:
            y = 1
        elif T[i] < 71:
            y = 2
        elif T[i] > 85:
            y = 3
        label[i, y] = 1
    return label


def resample(x, num, t=None, axis=0, window=None, domain='time'):
    """
    Emmanuel: 'Modified scipy resample for packaging purposes
            (to not require full installation of scipy)''
    source:
    https://github.com/scipy/scipy/blob/v1.11.4/scipy/signal/_signaltools.py#L3036-L3221

    Modified from scipy.
    Resample `x` to `num` samples using Fourier method along the given axis.
    fixed removed the complex option(cases)
    window  = None
    domain = always "time"
    t = None
    same as 'from scipy.signal import resample'
    """
    x = np.asarray(x)
    Nx = x.shape[axis]

    # Check if we can use faster real FFT
    real_input = np.isrealobj(x)
    X = np.fft.rfft(x, axis=axis)
    # Copy each half of the original spectrum to the output spectrum, either
    # truncating high frequences (downsampling) or zero-padding them
    # (upsampling)

    # Placeholder array for output spectrum
    newshape = list(x.shape)
    newshape[axis] = num // 2 + 1
    Y = np.zeros(newshape, X.dtype)

    # Copy positive frequency components (and Nyquist, if present)
    N = min(num, Nx)

    nyq = N // 2 + 1  # Slice index that includes Nyquist if present
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, nyq)
    Y[tuple(sl)] = X[tuple(sl)]

    # Split/join Nyquist component(s) if present
    # So far we have set Y[+N/2]=X[+N/2]
    if N % 2 == 0:
        if num < Nx:  # downsampling
            sl[axis] = slice(N//2, N//2 + 1)
            Y[tuple(sl)] *= 2.
        elif Nx < num:  # upsampling
            # select the component at frequency +N/2 and halve it
            sl[axis] = slice(N//2, N//2 + 1)
            Y[tuple(sl)] *= 0.5

    # Inverse transform
    y = np.fft.irfft(Y, num, axis=axis)
    y *= (float(num) / float(Nx))
    return y

# %% Load Dataset


def load_dataset(in_file1=BASE_DIR+"/Testing/"+Material+"/B_Field.csv",
                 in_file2=BASE_DIR+"/Testing/"+Material+"/Frequency.csv",
                 in_file3=BASE_DIR+"/Testing/"+Material+"/Temperature.csv"):

    data_B = np.genfromtxt(in_file1, delimiter=',')  # N by 1024, in T
    data_F = np.genfromtxt(in_file2, delimiter=',')  # N by 1, in Hz
    data_T = np.genfromtxt(in_file3, delimiter=',')  # N by 1, in C

    return data_B, data_F, data_T

# %% Calculate Core Loss


def core_loss(data_B, data_F, data_T):

    # ================ Wrap youcr model or algorithm here===============#

    # Here's just an example:

    length = len(data_F)
    data_P = np.random.rand(length, 1)

    # New stuff
    print(length)
    nparams = 24

    # Data transformation for model input
    T = OneHotEncoder(data_T)
    Bx = shift_data(data_B)  # start with smallest B
    B = resample(Bx, 24, axis=1)*1024/24
    B = np.log10(np.abs(B))

    F = np.log10(data_F).reshape(-1, 1)

    data_in = np.concatenate((F, B, T), axis=1)

    print("input size", data_in.shape)

    # ========================================
    # Model load
    paramaters = []
    modelSize = 0
    with open(scriptdir+"//"+Version+"//"+Material+".json", 'r') as f:
        parameters = json.load(f)
        for param in parameters:
            Ax = np.array(param)
            paramaters.append(Ax)
            modelSize += Ax.size
    print("Model size: ", modelSize)
    # Model parameters, Linear Layer weights and biases
    A1w = paramaters[0].T
    A1b = paramaters[1].reshape(1, -1)
    A2w = paramaters[2].T
    A2b = paramaters[3].reshape(1, -1)
    A3w = paramaters[4].T
    A3b = paramaters[5].reshape(1, -1)
    A4w = paramaters[6].T
    A4b = paramaters[7].reshape(1, -1)
    # Layer 1
    Bias = np.ones((data_in.shape[0], 1))
    outputs = np.matmul(data_in, A1w)+np.matmul(Bias, A1b)
    outputs = np.where(outputs > 0, outputs, 0)  # ReLU

    # Layer 2
    outputs = np.matmul(outputs, A2w)+np.matmul(Bias, A2b)
    outputs = np.where(outputs > 0, outputs, 0)  # ReLU
    # Layer 3
    outputs = np.matmul(outputs, A3w)+np.matmul(Bias, A3b)
    outputs = np.where(outputs > 0, outputs, 0)  # ReLU
    # Layer 4
    outputs = np.matmul(outputs, A4w)+np.matmul(Bias, A4b)

    outputs = 10**outputs  # antilog to get core loss in w/m^3

    data_P = outputs
    # ====================================================================#
    with open(BASE_DIR + "/Result/Volumetric_Loss_"
                       + Material+".csv", "w") as file:
        np.savetxt(file, data_P)
        file.close()

    print('Model inference is finished!')

    return

# %% Main Function for Model Inference


def main():

    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    print("Testing material {}".format(Material))

    data_B, data_F, data_T = load_dataset(
                 in_file1=BASE_DIR+"/Testing/"+Material+"/B_Field.csv",
                 in_file2=BASE_DIR+"/Testing/"+Material+"/Frequency.csv",
                 in_file3=BASE_DIR+"/Testing/"+Material+"/Temperature.csv")
    core_loss(data_B, data_F, data_T)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Accept input of the material "Material A" for example
        Material = sys.argv[1]

    main()

# End
