import torch 
import numpy as np
import random

from Model import MMINet
from Data import get_dataloader

# Target Material
Material = "Material E"

# Select GPU as default device
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Load Dataset
def load_dataset(in_file1="./Testing/"+Material+"/B_Field.csv", 
                 in_file2="./Testing/"+Material+"/Frequency.csv", 
                 in_file3="./Testing/"+Material+"/Temperature.csv"):
    
    data_B = np.genfromtxt(in_file1, delimiter=',') # N by 1024, in T
    data_F = np.genfromtxt(in_file2, delimiter=',') # N by 1, in Hz
    data_T = np.genfromtxt(in_file3, delimiter=',') # N by 1, in C

    return data_B, data_F, data_T

#%% Calculate Core Loss
def core_loss(data_B, data_F, data_T):
    
    #================ Wrap your model or algorithm here=======================#
    # 1.Create model isntances
    net = MMINet(Material).to(Device)   
    
    # 2.Load specific model 
    net_file="./"+Material+".pt"
    state_dict = torch.load(net_file, map_location=Device)
    net.load_state_dict(state_dict,strict=True)
    
    # 3.Get dataloader
    loader = get_dataloader(data_B,data_F,data_T,net.norm)
    
    # 4.Validate the models 
    data_P = torch.Tensor([]).to(Device)  # Allocate memory to store loss density
    
    with torch.no_grad():
        # Start model evaluation explicitly
        net.eval()

        for inputs, vars, s0 in loader: 
            Pv = net(inputs.to(Device), vars.to(Device), s0.to(Device)) 
            
            data_P = torch.cat((data_P,Pv),dim=0)
        print(data_P.shape)
    #=========================================================================#
    np.savetxt("./Testing/Result/Volumetric_Loss_"+Material+".csv", data_P.to('cpu'), delimiter=',')

    print('Model inference is finished!')
    
    return

#%% Main Function for Model Inference

def main():
    
    # Reproducibility
    MYSEED = 1
    random.seed(MYSEED)                         # Random seed
    np.random.seed(MYSEED)
    torch.manual_seed(MYSEED)                   # Data Loading
    torch.backends.cudnn.deterministic = True   # Deterministic operations
    torch.backends.cudnn.benchmark = False
    
    # Generate dataloader
    data_B, data_F, data_T = load_dataset()
    
    # Predict and save file 
    core_loss(data_B, data_F, data_T)
    
if __name__ == "__main__":
    main()
    
# End