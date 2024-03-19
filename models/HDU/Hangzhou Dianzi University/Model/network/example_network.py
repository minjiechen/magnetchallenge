import random
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import math
import os
import time


class Transformer(nn.Module):

    def __init__(self,
                 input_size: int,
                 dec_seq_len: int,
                 max_seq_len: int,
                 out_seq_len: int,
                 dim_val: int,
                 n_encoder_layers: int,
                 n_decoder_layers: int,
                 n_heads: int,
                 dropout_encoder,
                 dropout_decoder,
                 dropout_pos_enc,
                 dim_feedforward_encoder: int,
                 dim_feedforward_decoder: int,
                 dim_feedforward_projecter: int,
                 num_var: int = 2  # origin 3
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

        self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc,
                                                           max_len=max_seq_len)

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
        var = var.unsqueeze(1).repeat(1, enc_seq_len, 1)
        src = self.projector(torch.cat([src, var], dim=2))
        tgt = self.decoder_input_layer(tgt)
        tgt = self.positional_encoding_layer(tgt)
        tgt_mask = generate_square_subsequent_mask(sz1=self.out_seq_len, sz2=self.out_seq_len).to(device)
        output = self.decoder(
            tgt=tgt,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=None
        )
        output = self.linear_mapping(output)

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

# %% Main Function for Model Inference
