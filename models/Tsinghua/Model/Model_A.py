import random
import numpy as np
import torch
from torch import nn
import pandas as pd

Material = "Material A"


# %% Load Dataset

def load_dataset(in_file1="./Testing/" + Material + "/B_Field.csv",
                 in_file2="./Testing/" + Material + "/Frequency.csv",
                 in_file3="./Testing/" + Material + "/Temperature.csv"):
    data_B = np.genfromtxt(in_file1, delimiter=',')  # N by 1024, in T
    data_F = np.genfromtxt(in_file2, delimiter=',')  # N by 1, in Hz
    data_T = np.genfromtxt(in_file3, delimiter=',')  # N by 1, in C

    return data_B, data_F, data_T


# %% Calculate Core Loss
def core_loss(data_B, data_F, data_T):
    device = torch.device("cuda:0")
    BATCH_SIZE = 128
    # ================ Wrap your model or algorithm here=======================#


    data_F = data_F / np.max(data_F)
    data_T = data_T / np.max(data_T)

    Seq_tensors = torch.from_numpy(data_B).view(-1, 1024, 1)
    Freq_tensors = torch.from_numpy(data_F).view(-1, 1)
    Temperature_tensors = torch.from_numpy(data_T).view(-1, 1)
    test_dataset = torch.utils.data.TensorDataset(Seq_tensors, Freq_tensors, Temperature_tensors)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = torch.load('./Model_A_mixN87.pth')
    y_pred = []
    model.eval()
    with torch.no_grad():
        for seq, freq, temp in test_loader:
            y_pred.append(model(seq.to(device), freq.to(device), temp.to(device)))
    y_pred = torch.cat(y_pred, dim=0)
    data_P = 10 ** (y_pred.cpu().numpy())
    # yy_pred = pd.DataFrame(yy_pred)
    # yy_pred.to_csv('Material E Result_mixN87.csv')
    # =========================================================================#

    with open("./Testing/Volumetric_Loss_" + Material + ".csv", "w") as file:
        np.savetxt(file, data_P)
        file.close()

    print('Model inference is finished!')

    return


# %% Main Function for Model Inference

def main():
    # Reproducibility
    random.seed(1)
    np.random.seed(1)

    data_B, data_F, data_T = load_dataset()

    core_loss(data_B, data_F, data_T)


if __name__ == "__main__":
    main()

# End