import random
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import math
import os
import time

Material = "N49"

#%% Load Dataset

def load_dataset(in_file1="./Pretest Models/"+Material+"/B_waveform.csv", 
                 in_file2="./Pretest Models/"+Material+"/Frequency.csv", 
                 in_file3="./Pretest Models/"+Material+"/Temperature.csv"):
    
    data_B = np.genfromtxt(in_file1, delimiter=',')
    data_F = np.genfromtxt(in_file2, delimiter=',')
    data_T = np.genfromtxt(in_file3, delimiter=',')

    return data_B, data_F, data_T

#%% Post Processing

def post_processing(data_B, data_F, data_T,
                    norm_file="./Pretest Models/Norm_"+Material+".pt"):
    
    in_B = torch.from_numpy(data_B).float().view(-1, 1024, 1)
    in_F = torch.from_numpy(data_F).float().view(-1, 1)
    in_T = torch.from_numpy(data_T).float().view(-1, 1)
    in_D = torch.zeros_like(in_T)
    
    norm = torch.load(norm_file)
    
    in_B = (in_B[:,0::8,:]-norm[0])/norm[1]
    in_F = (torch.log10(in_F)-norm[2])/norm[3]
    in_T = (in_T-norm[4])/norm[5]
    in_D = (in_D-norm[6])/norm[7]
    normH = [norm[8],norm[9]]

    return torch.utils.data.TensorDataset(in_B, in_F, in_T, in_D), normH

#%% Define Model and Functions

class Transformer(nn.Module):

    def __init__(self, 
        input_size :int,
        dec_seq_len :int,
        max_seq_len :int,
        out_seq_len :int,
        dim_val :int,  
        n_encoder_layers :int,
        n_decoder_layers :int,
        n_heads :int,
        dropout_encoder,
        dropout_decoder,
        dropout_pos_enc,
        dim_feedforward_encoder :int,
        dim_feedforward_decoder :int,
        dim_feedforward_projecter :int,
        num_var: int=3
        ): 

        super().__init__() 

        self.dec_seq_len = dec_seq_len
        self.n_heads = n_heads
        self.out_seq_len = out_seq_len
        self.dim_val = dim_val

        self.encoder_input_layer = nn.Sequential(
            nn.Linear(input_size, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, dim_val))

        self.decoder_input_layer = nn.Sequential(
            nn.Linear(input_size, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, dim_val))

        self.linear_mapping = nn.Sequential(
            nn.Linear(dim_val, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, input_size))

        self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc, max_len=max_seq_len)
        
        self.projector = nn.Sequential(
            nn.Linear(dim_val + num_var, dim_feedforward_projecter),
            nn.Tanh(),
            nn.Linear(dim_feedforward_projecter, dim_feedforward_projecter),
            nn.Tanh(),
            nn.Linear(dim_feedforward_projecter, dim_val))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            activation="relu",
            batch_first=True
            )

        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=n_encoder_layers, norm=None)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            activation="relu",
            batch_first=True
            )

        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=n_decoder_layers, norm=None)

    def forward(self, src: Tensor, tgt: Tensor, var: Tensor, device) -> Tensor:

        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        src = self.encoder(src)
        enc_seq_len = 128
        var = var.unsqueeze(1).repeat(1,enc_seq_len,1)
        src = self.projector(torch.cat([src,var],dim=2))
        tgt = self.decoder_input_layer(tgt)
        tgt = self.positional_encoding_layer(tgt)
        tgt_mask = generate_square_subsequent_mask(sz1=self.out_seq_len, sz2=self.out_seq_len).to(device)
        output = self.decoder(
            tgt=tgt,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=None
            )
        output= self.linear_mapping(output)

        return output

class PositionalEncoder(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

def generate_square_subsequent_mask(sz1: int, sz2: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz1, sz2) * float('-inf'), diagonal=1)

#%% Main Function for Model Inference

def main():
    
    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    
    # Load Data and Conduct Post Processing
    data_B, data_F, data_T = load_dataset()
    dataset, normH = post_processing(data_B, data_F, data_T)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=5000, shuffle=False, **kwargs)
    
    # Load Model
    net = Transformer(
          dim_val=24,
          input_size=1, 
          dec_seq_len=129,
          max_seq_len=129,
          out_seq_len=129, 
          n_decoder_layers=1,
          n_encoder_layers=1,
          n_heads=4,
          dropout_encoder=0.0, 
          dropout_decoder=0.0,
          dropout_pos_enc=0.0,
          dim_feedforward_encoder=40,
          dim_feedforward_decoder=40,
          dim_feedforward_projecter=40).to(device)
    state_dict = torch.load("./Pretest Models/Model_"+Material+"_Transformer.sd")
    net.load_state_dict(state_dict, strict=True)
    
    # Model Inference
    with torch.no_grad():
        for in_B, in_F, in_T, in_D in test_loader:
            
            outputs = torch.zeros(in_B.size()[0], in_B.size()[1]+1, 1).to(device)
            tgt = (torch.rand(in_B.size()[0], in_B.size()[1]+1, 1)*2-1).to(device)
            tgt[:, 0, :] = 0.1*torch.ones(tgt[:, 0, :].size()).to(device)
                            
            for t in range(1, outputs.size()[1]):   
                outputs = net(src=in_B.to(device),tgt=tgt.to(device),var=torch.cat((in_F.to(device), in_T.to(device), in_D.to(device)), dim=1), device=device)                        
                tgt[:,t,:] = outputs[:,t-1,:]
            outputs = net(in_B.to(device),tgt.to(device),torch.cat((in_F.to(device), in_T.to(device), in_D.to(device)), dim=1), device=device)
            
            out_H = (outputs[:, :-1, :]*normH[1]+normH[0]).squeeze(2).cpu().numpy()
            
    loss = data_F * np.trapz(out_H, data_B[:,0::8], axis=1) 
    with open("./Pretest Results/pred_"+Material+".csv", "w") as f:
        np.savetxt(f, loss)
        f.close()

    print('Model inference is finished!')

if __name__ == "__main__":
    main()
    
