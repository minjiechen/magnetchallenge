import torch
import json
import numpy as np
from seq_processing import SeqDownsample, AddNoise, random_phase

def get_dataset(json_filename):    
    data_length=1024
##### Load json Files #####   
    with open(json_filename,'r') as load_f:           
        DATA = json.load(load_f)
    
    Btype = DATA['Waveform']          ####### label of waveform type
    Duty = DATA['Duty_cycle']         ####### duty cycle
    Slope = DATA['Slopes']            ####### slope
    Delta_B = DATA['Delta_B']         ####### delta B 
    Seq = DATA['B_waveform[T]']
    f = DATA['Frequency[Hz]']
    Temp=DATA['Temperature[C]']            ##### temperature

    ##### convert to label array
    max_length = max(len(sublist) for sublist in Duty)
    dutycycle = np.zeros((len(Duty), max_length))
    slope = np.zeros((len(Slope), max_length))
    deltaB = np.zeros((len(Delta_B), max_length))
    for i, sublist in enumerate(Duty):
        dutycycle[i, :len(sublist)] = sublist
    for i, sublist in enumerate(Slope):
        slope[i, :len(sublist)] = sublist
    for i, sublist in enumerate(Delta_B):
        deltaB[i, :len(sublist)] = sublist
    deltaB=np.abs(deltaB)*1000          
    Btype=np.array(Btype)
    Btype = np.where(Btype == "sine", 1, 0)    
    Seq = np.array(Seq)
    f=np.array(f)
    Temp = np.array(Temp)
    dutycycle=np.array(dutycycle)
    deltaB = np.array(deltaB)
 
    total_idx=np.arange(len(Seq))
    Seq = np.array([Seq[i] for i in total_idx])
    Temp = np.array([Temp[i] for i in total_idx])
    f = np.array([f[i] for i in total_idx])
    dutycycle = np.array([dutycycle[i] for i in total_idx])
    deltaB = np.array([deltaB[i] for i in total_idx])
    Btype=np.array([Btype[i] for i in total_idx])
    
    f_min = min(f)
    Seq = np.array(Seq)
    Seq = random_phase(Seq)      # randomly shift phase of the waveform
    Seq = SeqDownsample(Seq, f, f_min)   # downsampling of the waveform 
    Seq = AddNoise(Seq)  #add noise
    in_tensors = torch.from_numpy(Seq).view(-1, data_length, 1)   
    t = torch.from_numpy(Temp).view(-1, 1)                        
    Btype = torch.from_numpy(Btype).view(-1, 1)
    dutycycle=torch.from_numpy(dutycycle).view(-1, max_length, 1)

    ######### mean normalization ##########
    in_norm = [float(torch.mean(in_tensors)),float(torch.std(in_tensors))]
    t_norm = [float(torch.mean(t)),float(torch.std(t))]  
    in_tensors = (in_tensors-in_norm[0])/in_norm[1]
    t = (t-t_norm[0])/t_norm[1]                   
    deltaB1 = deltaB[:, 0]
    deltaB2 = deltaB[:, 1] 
    f=f/1000  #kHz
    logf=np.log10(f)
    logf=torch.from_numpy(logf).view(-1, 1)
    logBm=np.log10(deltaB1/2)
    logBm=torch.from_numpy(logBm).view(-1, 1)
    deltaB1=torch.from_numpy(deltaB1).view(-1, 1)
    deltaB2=torch.from_numpy(deltaB2).view(-1, 1)
    deltaB=torch.from_numpy(deltaB).view(-1, max_length, 1)
    
    ######### mean normalization ##########
    logf_norm = [float(torch.mean(logf)),float(torch.std(logf))]
    logBm_norm = [float(torch.mean(logBm)),float(torch.std(logBm))]
    deltaB1_norm = [float(torch.mean(deltaB1)),float(torch.std(deltaB1))] 
    logf = (logf-logf_norm[0])/logf_norm[1]
    logBm = (logBm-logBm_norm[0])/logBm_norm[1]
    deltaB1 = (deltaB1-deltaB1_norm[0])/deltaB1_norm[1]
    logf.requires_grad_(True)
    logBm.requires_grad_(True)
    deltaB1.requires_grad_(True)
    
    tensor_picked=torch.utils.data.TensorDataset(in_tensors,t,logf,logBm,deltaB1)
    return tensor_picked 