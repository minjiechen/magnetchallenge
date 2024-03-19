import random
import numpy as np
import utils
from utils import data_type_selection
import pandas as pd
import torch
from model_architectures import UNetGRU, MetaLMgenerator, MetaLMlayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Material = "Material E"

spike_threshold = {"Material A":0.000156, "Material B":0.000148, "Material C":0.0001725, "Material D":0.000177, "Material E":0.00015}

def load_dataset(in_file1="./Testing/"+Material+"/B_Field.csv", 
                 in_file2="./Testing/"+Material+"/Frequency.csv", 
                 in_file3="./Testing/"+Material+"/Temperature.csv"):
    
    data_B = np.array(pd.read_csv(in_file1, header=None)) # N by 1024, in T
    data_F = np.array(pd.read_csv(in_file2, header=None)) # N by 1, in Hz
    data_T = np.array(pd.read_csv(in_file3, header=None)) # N by 1, in C

    data_length = 1024
    data_B = torch.from_numpy(data_B).float().view(-1, data_length, 1)
    data_F = torch.from_numpy(data_F).float().view(-1, 1)
    data_T = torch.from_numpy(data_T).float().view(-1, 1)
    in_B_type = data_type_selection(data_B, threshold=spike_threshold[Material])

    norm = [torch.mean(data_B), torch.std(data_B), torch.mean(data_F), torch.std(data_F)]
    data_B = (data_B-torch.mean(data_B))/torch.std(data_B)
    data_F = (data_F-torch.mean(data_F))/torch.std(data_F)
    data_T = (data_T-torch.mean(data_T))/torch.std(data_T)

    return data_B, data_F, data_T, in_B_type, norm


def core_loss(data_B, data_F, data_T, in_B_type, norm):
    
    #================ Wrap your model or algorithm here=======================#
    
    dataset = torch.utils.data.TensorDataset(data_B, data_F, data_T, in_B_type)
    loader = torch.utils.data.DataLoader(dataset, batch_size = 128, shuffle=False)
    #================ Wrap your model or algorithm here=======================#
    
    net = torch.load('content/Model_'+Material+'_distilled.pkl', map_location= device)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))
    
    H_list= []
    B_denorm= []
    F_denorm= []
    
    for i, (B, F, T, B_type) in enumerate(loader):
        input_signal = B.to(device)
        meta_data = torch.cat((F, T, B_type), dim=1).to(device)

        outputs = net(input_signal,meta_data)
        
        normH = torch.load('content/Unet_norm_'+Material+'.pt')
        
        H_list.append((outputs*normH[1]+normH[0]).squeeze(2).detach().cpu().numpy())
        B_denorm.append((B*norm[1]+norm[0]).squeeze(2).numpy())
        F_denorm.append((F*norm[3]+norm[2]).squeeze())

    H_list = np.concatenate(H_list)
    data_B = np.concatenate(B_denorm)
    data_F_real = np.concatenate(F_denorm)
    vol_loss = data_F_real * np.trapz(H_list, data_B, axis=1) 
    #=========================================================================#
    
    with open("./Results/Volumetric_Loss_"+Material+".csv", "w") as file:
        np.savetxt(file, vol_loss)
        file.close()
    
    print('Model inference is finished!')
    
    return


def main():
    
    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    data_B, data_F, data_T, in_B_type, norm = load_dataset()
    print(data_B.shape)
    core_loss(data_B, data_F, data_T, in_B_type, norm)
    
if __name__ == "__main__":
    main()
    
