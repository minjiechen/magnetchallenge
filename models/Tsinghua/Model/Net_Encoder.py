import torch
from torch import nn
import numpy as np
import pandas as pd
from torch import Tensor
import math
class Encoder(nn.Module):
    def __init__(self,input_size :int,max_seq_len :int,
                 dim_val: int,
                 n_encoder_layers: int,
                 n_heads: int,
                 dropout_encoder,
                 dim_feedforward_encoder: int,
                 dropout_pos_enc
                 ):
        super(Encoder,self).__init__()
        self.n_heads = n_heads
        self.dim_val = dim_val
        self.encoder_input_layer = nn.Sequential(
            nn.Linear(input_size, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, dim_val))
        self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc,
                                                           max_len=max_seq_len)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            activation="relu",
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=n_encoder_layers, norm=None)
        self.lstm = nn.LSTM(64, 64, num_layers=1, batch_first=True, bidirectional=False)
        self.fc1 = nn.Sequential(
        nn.Linear(64, 32),
        nn.LeakyReLU(0.2),

        nn.Linear(32, 16),
        nn.LeakyReLU(0.2),

        nn.Linear(16, 14))
        self.fc2 = nn.Sequential(
        nn.Linear(16, 16),
        nn.LeakyReLU(0.2),
        nn.Linear(16, 8),
        nn.LeakyReLU(0.2),
        nn.Linear(8, 1)
        )
        # self.fc0 = nn.Sequential(
        #      nn.Linear(1024,512),
        #      nn.GELU(),
        #      nn.Linear(512,206),
        #      nn.GELU(),
        #      nn.Linear(206,50),
        #      nn.GELU(),
        #      nn.Linear(50,1),
        #      nn.GELU()
        #      )
    def forward(self,src:Tensor,freq,temp) -> Tensor:
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        x = self.encoder(src)
        #x = self.fc0(x.view(x.shape[0], 32, 1024))
        #x = x.view(x.shape[0],32)
        x,_ = self.lstm(x)
        x = x[:,-1,:]
        x = self.fc1(x)
        y = self.fc2(torch.cat((x, freq, temp), 1))
        return y

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
    #Generates an upper-triangular matrix of -inf, with zeros on diag.
    return torch.triu(torch.ones(sz1, sz2) * float('-inf'), diagonal=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

