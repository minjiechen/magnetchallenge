import random
import numpy as np
import torch
import csv
from model_definition import Net,count_parameters
from get_dataset import get_dataset
from classifier import classifier

Material = 'Material A'
#%% Load Dataset

def load_dataset(in_file1="../"+Material+"/B_Field.csv", 
                 in_file2="../"+Material+"/Frequency.csv", 
                 in_file3="../"+Material+"/Temperature.csv"):
    
    data_B = np.genfromtxt(in_file1, delimiter=',') # N by 1024, in T
    data_F = np.genfromtxt(in_file2, delimiter=',') # N by 1, in Hz
    data_T = np.genfromtxt(in_file3, delimiter=',') # N by 1, in C
    return data_B, data_F, data_T


#%% Calculate Core Loss
def core_loss(data_B, data_F, data_T):
    
    #================ Wrap your model or algorithm here=======================#
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    
    # Hyperparameters
    NUM_EPOCH = 500
    BATCH_SIZE = 256
    DECAY_EPOCH = 100
    DECAY_RATIO = 0.5
    LR_INI = 0.01
    Hyperparameters_list = [NUM_EPOCH, BATCH_SIZE, DECAY_EPOCH, DECAY_RATIO, LR_INI]
    
    # Load Model
    load_model="../Model_"+Material+".sd"
    lstm_hidden_size,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4=32,12,12,12,12
    net = Net(lstm_hidden_size,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4).double().to(device)  
    state_dict = torch.load(load_model)    
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    print("Model is loaded!")
    
    ###### Load Data and Conduct data Processing
    ###### convert dataset to labeled json file 
    classifier(data_B,data_F,data_T,"../labeled_"+Material+".json" ,"../labeled_"+Material+".json" )   ### jump this step if already generate labeled json file
    json_filename = "../labeled_"+Material+".json"   
    print("Labeled json file is generated")
    
    # Model Inference            
    ###### Load dataset ######  
    test_dataset = get_dataset(json_filename)
    test_size = len(test_dataset)
    ###### Log the number of parameters ######
    print("Number of parameters: ", count_parameters(net))
    print("test size: ",test_size)
    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    ####### Evaluation #######  
    norm_file ="../Norm_"+Material+".pt" 
    norm = torch.load(norm_file)
    net.eval()   
    y_pred = []
    with torch.no_grad():  
        for inputs,t,logf,logBm,deltaB1 in test_loader:    
            y_pred.append(net(inputs.to(device), t.to(device),logf.to(device),logBm.to(device),deltaB1.to(device)))
    y_pred = torch.cat(y_pred, dim=0)
    yy_pred = 10**((y_pred.cpu().numpy())*norm[1]+norm[0])*1000
    predict = yy_pred.squeeze(1)
    with open("../../Result/pred_"+Material+".csv",'w',newline = '') as csvfile:
        csvwriter = csv.writer(csvfile)               
        for i in range(len(predict)):
            csvwriter.writerow([predict[i]]) 
    print('Model inference is finished!')
    return

#%% Main Function for Model Inference

def main():
    # Reproducibility
    random.seed(96)
    np.random.seed(96)
    data_B, data_F, data_T = load_dataset()
    core_loss(data_B, data_F, data_T)
    
if __name__ == "__main__":
    main()
    
# End