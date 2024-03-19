import torch
import torch.nn as nn
import random
import numpy as np


Material = "Material A"
#%% Load Dataset
def load_dataset(in_file1="../Testing/"+Material+"/B_Field.csv", 
                 in_file2="../Testing/"+Material+"/Frequency.csv", 
                 in_file3="../Testing/"+Material+"/Temperature.csv"):
    
    data_B = np.genfromtxt(in_file1, delimiter=',') # N by 1024, in T
    data_F = np.genfromtxt(in_file2, delimiter=',') # N by 1, in Hz
    data_T = np.genfromtxt(in_file3, delimiter=',') # N by 1, in C

    return data_B, data_F, data_T

#%% Calculate Core Loss
def core_loss(data_B, data_F, data_T):
    #================ Wraped data pre-processing and model ===================#
    #++++++++ FFT of the waveform to extract frequency domain info. ++++++++#
    ## Total harmonic oder
    n = 3
    N = np.arange(1, n+1)
    # Initialize arrays to store Fourier coefficients for each sample
    num_samples = data_B.shape[0]
    Fourier_coeffs_magnitudes = np.zeros((num_samples, n))
    Fourier_coeffs_phases = np.zeros((num_samples, n))
    Fourier_freq = np.zeros((num_samples, n))  # Array to store frequencies
    ## Carry out FFT to obtain the frequency domain info.
    for row in range(num_samples):
        ### wave length and sample rate
        B = data_B[row]
        f0 = data_F[row]
        Fs = (len(B)-1) * f0
        L = len(B)
        ### FFT
        B_fft = np.fft.fft(B)/L
        B_fft = B_fft[0:int(L/2)]
        freq = np.fft.fftfreq(L)*Fs
        freq = freq[0:int(L/2)]
        B_mg = 2*np.abs(B_fft)
        B_ph = np.angle(B_fft,deg=True)
        # Store Fourier coefficients for this row
        Fourier_coeffs_magnitudes[row, :] = B_mg[N]
        Fourier_coeffs_phases[row, :] = B_ph[N]
        Fourier_freq[row, :] = f0 * N  # Store the frequencies corresponding to each harmonic
    ## Log transformation
    Fourier_coeffs_magnitudes_log = np.log10(Fourier_coeffs_magnitudes + 1e-9)   # Add a small constant to avoid log(0)
    Fourier_freq_log = np.log10(Fourier_freq + 1e-9)
    ## reshape the data
    in_mag_flat_test = torch.from_numpy(Fourier_coeffs_magnitudes_log).float().view(-1, n)
    in_f_flat_test = torch.from_numpy(Fourier_freq_log).float().view(-1, n)
    in_T_flat_test = torch.from_numpy(data_T).float().view(-1, 1)
    #++++++++++++++++++++++++ End of the FFT ++++++++++++++++++++++++#

    #++++++++ Normalize the data using the same transformation applied to the training dataset ++++++++#
    ## load the normalization parameters used for the normalization of the training data
    load_path = "./"+Material+" norm params.pth"
    norm_params = torch.load(load_path)
    normout_mag, normout_f, normout_T, normout_loss = norm_params
    ## Normalize each feature using the normalization parameters from the training data
    in_mag_norm = (in_mag_flat_test - normout_mag[0]) / normout_mag[1]
    in_f_norm = (in_f_flat_test - normout_f[0]) / normout_f[1]
    in_T_norm = (in_T_flat_test - normout_T[0]) / normout_T[1]
    ## Expand dimensions
    in_mag_exp = in_mag_norm.unsqueeze(-1)
    in_f_exp = in_f_norm.unsqueeze(-1)
    ## Stack and reshape features
    harmonic_features_stacked = torch.stack((in_f_exp, in_mag_exp), dim=2)
    harmonic_features_flat = harmonic_features_stacked.view(-1, 2 * n)
    ## Concatenate with in_T_norm
    input_vector_bundled = torch.cat((harmonic_features_flat, in_T_norm), dim=1)
    ## Convert to double
    input_vector_bundled = input_vector_bundled.double()
    assert isinstance(input_vector_bundled, torch.Tensor), "input_vector_bundled must be a torch.Tensor"
    test_dataset = torch.utils.data.TensorDataset(input_vector_bundled)
    #++++++++++++++++++++++++ End of the normalization of the test data ++++++++++++++++++++++++#

    #++++++++++++++++++++++++++ Prepare the Neural Network ++++++++++++++++++++++++++++++++#
    ## Net hyperparameters for the material
    batch_size =64
    num_layers = 3
    neurons_per_layer = [30,18,31]
    activation_function = nn.SiLU()  
    ## Create the Net
    class Net(nn.Module):
        def __init__(self, num_layers, neurons_per_layer, activation):
            super(Net, self).__init__()
            layers = []
            ### Input layer
            layers.append(nn.Linear(7, neurons_per_layer[0]))
            layers.append(activation)
            ### Hidden layers
            for i in range(1, num_layers):
                layers.append(nn.Linear(neurons_per_layer[i-1], neurons_per_layer[i]))
                layers.append(activation)
            ### Output layers
            layers.append(nn.Linear(neurons_per_layer[-1], 1))
            ### Compose the layers
            self.layers = nn.Sequential(*layers)
        ### Define the forward path
        def forward(self, x):
            return self.layers(x)
    ## Counting the parameter numbers    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    #++++++++++++++++++++++++ End of preparing the Neural Network ++++++++++++++++++++++++#

    #++++++++++++++++++++++++ Loss prediction ++++++++++++++++++++++++++++++++#
    ## Prepare the dataloader
    ### Define the dataloader parameters
    kwargs = {'num_workers': 0, 'pin_memory': True, 'pin_memory_device': "cuda"}
    ### Create the test dataloader 
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    ## Select GPU as default device
    device = torch.device("cuda")
    ## Setup network
    net = Net(num_layers, neurons_per_layer, activation_function).double().to(device)
    print("Total paramerter number is", count_parameters(net))
    ## Load the trained net weights and bias 
    load_path = "./Model "+Material+".sd"
    net.load_state_dict(torch.load(load_path))

    ## Start the loss prediction
    ### Ensure the model is in evaluation mode
    net.eval()
    y_pred = []
    with torch.no_grad():
        for batch in test_loader: # feed the test dataloader to the net with the predefined batch size
            input_vector_bundled = batch[0]  
            y_pred.append(net(input_vector_bundled.to(device)))
    y_pred = torch.cat(y_pred, dim=0) 
    ### Denormalize the output using the same transformation applied to the trainind data
    y_pred_denorm = (y_pred.cpu() * normout_loss[1]) + normout_loss[0]
    ### Inverse of log to obtain the predicted loss
    data_P = 10**(y_pred_denorm.numpy())
    #+++++++++++++++++++++ Ens of the loss prediction +++++++++++++++++++++++++#
    #=========================================================================#
    
    with open("../Result/Volumetric_Loss_"+Material+".csv", "w") as file:
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