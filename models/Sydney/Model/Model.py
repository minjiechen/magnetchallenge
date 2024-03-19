import torch 
import numpy as np
import torch.nn as nn
from torch.autograd import Variable 

# Material normalization data (1.B 2.H 3.F 4.T 5.dB/dt 6.Pv)
normsDict = {"Material A": [[-4.02296069e-19,  6.42790612e-02],
                            [ 1.15118525e-01,  1.22041107e+01],
                            [ 5.16368866e+00,  2.68540382e-01],
                            [ 5.52569885e+01,  2.61055470e+01],
                            [ 2.42224485e-01,  2.37511802e+00],
                            [ 4.94751596e+00,  8.27844262e-01]], 
             "Material B": [[ 6.75135623e-20,  6.27030179e-02],
                            [ 3.95575739e-02,  7.62486081e+00],
                            [ 5.26432657e+00,  2.88519919e-01],
                            [ 5.80945930e+01,  2.40673885e+01],
                            [ 2.72521585e-01,  2.46433449e+00],
                            [ 5.05083704e+00,  7.10303366e-01]], 
             "Material C": [[-7.61633305e-19,  7.95720905e-02],
                            [ 1.11319124e-01,  1.30629103e+01],
                            [ 5.18559408e+00,  2.68714815e-01],
                            [ 5.84123573e+01,  2.40717468e+01],
                            [ 3.26634765e-01,  3.03949690e+00],
                            [ 4.74633312e+00,  8.05532336e-01]],
             "Material D": [[-3.82835526e-18,  8.10498434e-02],
                            [-1.14488902e-02,  2.83868927e+01],
                            [ 5.25141287e+00,  2.50821203e-01],
                            [ 6.72413788e+01,  2.59518223e+01],
                            [ 3.00584078e-01,  3.24369454e+00],
                            [ 5.01819372e+00,  8.41059685e-01]], 
             "Material E": [[-4.22607249e-18,  1.28762770e-01],
                            [ 3.88389004e-01,  4.80431443e+01],
                            [ 5.18909550e+00,  2.77695119e-01],
                            [ 5.64505730e+01,  2.46127701e+01],
                            [ 6.35038793e-01,  5.19237566e+00],
                            [ 5.68955612e+00,  7.26979315e-01]]}

# %% Magnetization mechansim-determined neural network
class MMINet(nn.Module):

    """
     Parameters:
      - hidden_size: number of eddy current slices (RNN neuron)
      - operator_size: number of operators
      - input_size: number of inputs (1.B 2.dB 3.dB/dt)
      - var_size: number of supplenmentary variables (1.F 2.T)        
      - output_size: number of outputs (1.H)
    """

    def __init__(self, Material, hidden_size=30, operator_size=30,
                  input_size=3,var_size=2,output_size=1):
        super().__init__()   
        self.input_size = input_size
        self.var_size = var_size
        self.hidden_size = hidden_size
        self.output_size = output_size 
        self.operator_size = operator_size
        self.norm = normsDict[Material]           # normalization data

        # Consturct the network 
        self.rnn1 = StopOperatorCell(self.operator_size)
        self.dnn1 = nn.Linear(self.operator_size+2,1)
        self.rnn2 = EddyCell(4,self.hidden_size,output_size)
        self.dnn2 = nn.Linear(self.hidden_size,1)

        self.rnn2_hx = None

    def forward(self, x, var, s0, n_init=16):
        """
         Parameters: 
          - x(batch,seq,input_size): Input features (1.B, 2.dB, 3.dB/dt)  
          - var(batch,var_size): Supplementary inputs (1.F 2.T)
          - s0(batch,1): Operator inital states
        """    
        batch_size = x.size(0)          # Batch size 
        seq_size = x.size(1)            # Ser
        self.rnn1_hx = s0
        
        # Initialize DNN2 input (1.B 2.dB/dt)
        x2 = torch.cat((x[:,:,0:1],x[:,:,2:3]),dim=2)
        
        for t in range(seq_size):
            # RNN1 input (dB,state)       
            self.rnn1_hx = self.rnn1(x[:,t,1:2], self.rnn1_hx)

            # DNN1 input (rnn1_hx,F,T)
            dnn1_in = torch.cat((self.rnn1_hx,var),dim=1) 

            # H hysteresis prediction 
            H_hyst_pred = self.dnn1(dnn1_in)

            # DNN2 input (B,dB/dt,T,F)
            rnn2_in = torch.cat((x2[:,t,:],var),dim=1) 

            # Initialize second rnn state 
            if t==0:
                H_eddy_init = x[:,t,0:1]-H_hyst_pred
                buffer = x.new_ones(x.size(0),self.hidden_size)
                self.rnn2_hx = Variable((buffer/torch.sum(self.dnn2.weight,dim=1))*H_eddy_init)

            #rnn2_in = torch.cat((rnn2_in,H_hyst_pred),dim=1)
            self.rnn2_hx = self.rnn2(rnn2_in, self.rnn2_hx)

            # H eddy prediction
            H_eddy = self.dnn2(self.rnn2_hx)

            # H total 
            H_total = (H_hyst_pred+H_eddy).view(batch_size,1,self.output_size)
            if t==0:
                output = H_total
            else:
                output = torch.cat((output,H_total),dim=1)
                
        # Compute the power loss density 
        B = (x[:,n_init:,0:1]*self.norm[0][1]+self.norm[0][0]) 
        H = (output[:,n_init:,:]*self.norm[1][1]+self.norm[1][0]) 
        Pv = torch.trapz(H,B,axis=1)*(10**(var[:,0:1]*self.norm[2][1]+self.norm[2][0]))
        
        return torch.flatten(Pv)
    
# %% MMINN Sub-layer: Static hysteresis prediction using stop operators 
class StopOperatorCell():

    """ 
      Parameters:
      - operator_size: number of operator
    """

    def __init__(self, operator_size):
        self.operator_thre = torch.from_numpy(np.linspace(5/operator_size
                            ,5
                            ,operator_size)).view(1,-1)
        
    def sslu(self, X):
        """ Hardsimoid-like or symmetric saturated linear unit definition

        """
        a = torch.ones_like(X)
        return torch.max(-a,torch.min(a,X))
    
    def __call__(self, dB, state):
        """ Update operator of each time step

        """
        r = self.operator_thre.to(dB.device) 
        output = self.sslu((dB + state)/r)*r
        return output.float()
  
# %% MMINN subsubnetwork: Dynamic hysteresis prediction
class EddyCell(nn.Module):

    """ 
      Parameters:
      - input_size: feature size 
      - hidden_size: number of hidden units (eddy current layers)
      - output_size: number of the output
    """

    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.x2h = nn.Linear(input_size, hidden_size ,bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size ,bias=False)

    def forward(self, x, hidden=None):
        """
         Parameters:
         - x(batch,input_size): features (1.B 2.dB/dt 3.F 4.T)
         - hidden(batch,hidden_size): dynamic hysteresis effects at each eddy current layer 
        """
        hidden = self.x2h(x) + self.h2h(hidden)
        hidden = torch.sigmoid(hidden)
        return hidden
