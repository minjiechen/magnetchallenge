import torch
import numpy as np

# %% Preprocess data into a data loader 
def get_dataloader(data_B,data_F,data_T,norm,
                   n_init = 16):
    """Get a test dataloader 

    Parameters
    ---------
    data_B/data_F/data_T : array 
         B/F/T data
    norm : list 
         B/F/T normalization data
    n_init : int
         Additional points for computing the history magnetization
    """
    
    # Data pre-process 
    # 1. Down-sample to 128 points 
    seq_length = 128
    cols = range(0,1024,int(1024/seq_length))
    data_B = data_B[:,cols]

    # 2. Add extra points for initial magnetization calculation 
    data_length = seq_length + n_init
    data_B = np.hstack((data_B,data_B[:,:n_init]))
    
    # 3. Format data into tensors 
    B = torch.from_numpy(data_B).view(-1,data_length,1).float()
    F = torch.log10(torch.from_numpy(data_F).view(-1,1).float())
    T = torch.from_numpy(data_T).view(-1,1).float()
    
    # 4. Data Normalization 
    in_B = (B-norm[0][0])/norm[0][1]
    in_T = (T-norm[3][0])/norm[3][1]
    in_F = (F-norm[2][0])/norm[2][1]
    
    # 5. Extra features 
    dB = torch.diff(B,dim=1)
    dB = torch.cat((dB[:,0:1],dB),dim=1)
    dB_dt = dB*(seq_length*F.view(-1,1,1))
    
    in_dB = torch.diff(B,dim=1)                     # Flux density change
    in_dB = torch.cat((in_dB[:,0:1],in_dB),dim=1)
    
    in_dB_dt = (dB_dt-norm[4][0])/norm[4][1]        # Flux density change rate
    
    max_B,_ = torch.max(in_B,dim=1)
    min_B,_ = torch.min(in_B,dim=1)
    
    s0 = get_operator_init(in_B[:,0]-in_dB[:,0], in_dB, max_B, min_B) # Operator inital state 
    
    # 6. Create dataloader to speed up data processing
    test_dataset = torch.utils.data.TensorDataset(torch.cat((in_B,in_dB,in_dB_dt),dim=2), torch.cat((in_F, in_T),dim=1), s0)
    kwargs = {'num_workers': 4, 'batch_size':128, 
                        'pin_memory': True, 'pin_memory_device': "cuda" if torch.cuda.is_available() else "cpu",
                        'drop_last': False}
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

    return test_loader

# %% Predict the operator state at t0
def get_operator_init(B1, dB, Bmax, Bmin,
                      max_out_H = 5,
                      operator_size = 30):
    """Compute the inital state of hysteresis operators

    Parameters
    ---------
    B1 : torch_like (batch)
         Stop operator excitation at t1
    dB : torch_like (batch, data_length)
         Flux density changes at each t
    Bmax/Bmin : torch_like (batch)
         Max/Min flux density of each cycle 
    """
    # 1. Parameter setting
    s0 = torch.zeros((dB.shape[0], operator_size))        # Allocate cache for s0
    operator_thre = torch.from_numpy(np.linspace(max_out_H/operator_size
                        ,max_out_H
                        ,operator_size)).view(1,-1) # hysteresis operators' threshold

    # 2. Iterate each excitation for the operator inital state computation
    for i in range(dB.shape[0]):
        for j in range(operator_size):
            r = operator_thre[0,j]
            if (Bmax[i] >= r) or (Bmin[i] <= -r):
                if dB[i,0] >= 0:
                    if B1[i] > Bmin[i]+2*r:
                        s0[i,j] = r
                    else:
                        s0[i,j] = B1[i]-(r+Bmin[i])
                else:
                    if B1[i] < Bmax[i]-2*r:
                        s0[i,j] = -r
                    else:
                        s0[i,j] = B1[i]+(r-Bmax[i])

    return s0