import random
import numpy as np

Material = "Material A"

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
    
    # Here's just an example:
    
    length = len(data_F)
    data_P = np.random.rand(length,1)
    
    #=========================================================================#
    
    with open("./Testing/Volumetric_Loss_"+Material+".csv", "w") as file:
        np.savetxt(file, data_P)
        file.close()

    print('Model inference is finished!')
    
    return

#%% Main Function for Model Inference

def main():
    
    # Reproducibility
    random.seed(1)
    np.random.seed(1)

    data_B, data_F, data_T = load_dataset()
    
    core_loss(data_B, data_F, data_T)
    
if __name__ == "__main__":
    main()
    
# End