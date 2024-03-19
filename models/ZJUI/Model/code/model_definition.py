import torch.nn as nn
import torch
import numpy as np

class Net(nn.Module):   # define network structure
    def __init__(self,lstm_hidden_size,hidden_size_1, hidden_size_2, hidden_size_3,hidden_size_4):
        super(Net, self).__init__()

        self.lstm = nn.GRU(1, lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=False)  
# hidden size(num of features in hidden states)
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_hidden_size+4, hidden_size_1),       ############## add 4 neurons for temperature, f, B1,B2
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(),
            nn.Linear(hidden_size_3, hidden_size_4),
            nn.ReLU(),
            nn.Linear(hidden_size_4, 1)
        )
        for m in self.modules():       #######initialization 
            if isinstance(m, torch.nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, x,y,logf,logBm,deltaB1):            
        x, _ = self.lstm(x)
        x = x[:, -1, :] 
        x = x.squeeze(0)
        z = self.fc_layers(torch.cat([x,y,logf,logBm,deltaB1],dim=1)) 
        return z

#%% count number of parameters of neural network
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
