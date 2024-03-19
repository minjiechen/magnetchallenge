import random
import numpy as np
# modified
import torch
import torch.nn as nn

Material = "Material E"

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
    # Net Class
    class Net(nn.Module):
        def __init__(self, load_pretrained: bool = False, pretrained_model_path :str  = "None"):
            super(Net, self).__init__()
            # Define a fully connected layers model with three inputs (frequency, flux density, duty ratio) and one output (power loss).
            self.layers = nn.Sequential(
                nn.Linear(1026, 65),
                nn.ReLU(),
                nn.Linear(65, 55),
                nn.ReLU(),
                nn.Linear(55, 116),
                nn.ReLU(),
                nn.Linear(116, 40),
                nn.ReLU(),
                nn.Linear(40, 123),
                nn.ReLU(),
                nn.Linear(123, 1),
            )
            if load_pretrained and pretrained_model_path is not None:
                self.load_pretrained_model(pretrained_model_path)

        def forward(self, x):
            return self.layers(x)

        def load_pretrained_model(self, path):
            pretrained_dict = torch.load(path)
            model_dict = self.state_dict()
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print('Model is load')
    
    def load_dataset(B,Freq,Temp):
        # B = read_csv(B_file_path)
        # Freq = read_csv(Freq_file_path)
        # Temp = read_csv(Temp_file_path)
        # #H = read_csv(H_file_path)
        # Power = read_csv(Power_file_path)

        # Compute labels
        # There's approximalely an exponential relationship between Loss-Freq and Loss-Flux.
        # Using logarithm may help to improve the training.
        Freq = np.log10(Freq)
        Temp = np.array(Temp)
        # Power = np.log10(Power)

        # Reshape data
        Freq = torch.from_numpy(Freq).float().view(-1, 1)
        B = torch.from_numpy(B).float().view((-1,1024,1))
        #H = torch.from_numpy(H).float().view((-1,1024,1))
        Temp = torch.from_numpy(Temp).view(-1, 1)
        # Power = Power.reshape((-1,1))

        # Normalize
        B = (B-torch.mean(B))/torch.std(B).numpy()
        #H = (H-torch.mean(H))/torch.std(H).numpy()
        Freq = (Freq-torch.mean(Freq))/torch.std(Freq).numpy()
        Temp = (Temp-torch.mean(Temp))/torch.std(Temp).numpy()

        B = np.squeeze(B, axis=2)
        #H = np.squeeze(H, axis=2)

        print(np.shape(Freq))
        print(np.shape(B))
        #print(np.shape(H))
        print(np.shape(Temp))
        # print(np.shape(Power))

        temp = np.concatenate((Freq,B,Temp),axis=1)

        in_tensors = torch.from_numpy(temp).view(-1, 1026)
        out_tensors = torch.empty_like(in_tensors)
        #out_tensors = torch.from_numpy(Power).view(-1, 1)

        #return torch.utils.data.TensorDataset(in_tensors, out_tensors) 
        return torch.utils.data.TensorDataset(in_tensors, out_tensors) 

    # Load trained parameters
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hyperparameters
    BATCH_SIZE = 128

    # Select GPU as default device
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    # Load Dataset
    test_dataset = load_dataset(data_B,data_F,data_T)
    if cuda_available:
        kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    else:
        kwargs = {'num_workers': 0, 'pin_memory': False, 'pin_memory_device': "cpu"}

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    net = Net().double().to(device)
    
    state_dict = torch.load(f'./models/Model_{Material}.sd',map_location=device)
    net.load_state_dict(state_dict, strict=True)
    net.eval()
    print("Model is loaded!")

    # y_meas = []
    y_pred = []
    with torch.no_grad():
        # for inputs, labels in test_loader:
        for inputs,labels in test_loader:
            y_pred.append(net(inputs.to(device)))
            # y_meas.append(labels.to(device))
    
    # y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
   #print(f"Test Loss: {F.mse_loss(y_meas, y_pred).item() / len(test_dataset) * 1e5:.5f}")

    yy_pred = 10**(y_pred.cpu().numpy())
    # yy_meas = 10**(y_meas.cpu().numpy())
    print("yy_pred :", yy_pred.shape)
    # print("yy_meas :", yy_meas.shape)

    data_P = yy_pred
    
    # # Output Test Performance
    # # Error_re = abs(yy_pred-yy_meas)/abs(yy_meas)*100
    # # Error_re_avg = np.mean(Error_re)
    # Error_re_rms = np.sqrt(np.mean(Error_re ** 2))
    # Error_re_95prct = np.percentile(Error_re, 95)
    # Error_re_99prct = np.percentile(Error_re, 99)
    # Error_re_max = np.max(Error_re)

    # print(f"Relative Error: {Error_re_avg:.8f}")
    # print(f"AVG Error: {Error_re_avg:.8f}")
    # # print(f"RMS Error: {Error_re_rms:.8f}")
    # print(f"95-PRCT Error: {Error_re_95prct:.8f}")
    # print(f"99th Percentile Error: {Error_re_99prct:.8f}")
    # print(f"MAX Error: {Error_re_max:.8f}")

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