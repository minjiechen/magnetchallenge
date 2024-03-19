import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from utils.util import train_valid_target_eval_names, alg_loss_dict
from network import Adver_network
import argparse
from datautil.getdataloader import get_LSTM_dataloader


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


class EncoderAttention(nn.Module):
    def __init__(self,
                 input_size: int,
                 max_seq_len: int,
                 dim_val: int,
                 n_encoder_layers: int,
                 n_heads: int,
                 dropout_encoder,
                 dropout_pos_enc,
                 dim_feedforward_encoder: int
                 ):
        super(EncoderAttention, self).__init__()

        self.n_heads = n_heads
        self.dim_val = dim_val

        self.encoder_input_layer = nn.Sequential(
            nn.Linear(input_size, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, dim_val))

        self.pos_encoder = PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc,
                                             max_len=max_seq_len)
        # self.pos_encoder = PositionalEncoder(d_model=input_size, dropout=dropout_pos_enc,
        #                                      max_len=max_seq_len)



        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            activation="relu",
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=n_encoder_layers,
                                                         norm=None)

    def forward(self, src: Tensor) -> Tensor:
        src = self.encoder_input_layer(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)

        return output


class Att2DANN(nn.Module):
    # wip
    def __init__(self, attention_module, projector_head, projector_regressor, projector_discriminator):
        super(Att2DANN, self).__init__()

        self.att_encoder = attention_module
        self.featurizer = projector_head
        self.regressor = projector_regressor
        self.domain_classifier = projector_discriminator
        self.alpha = 1  # default 1

    def update(self, minibatches, opt, sch):
        all_bh = torch.cat([data[6].cuda().float() for data in minibatches])
        # all_freq = torch.cat([data[2].cuda().float() for data in minibatches])
        all_freq = torch.cat([data[9].cuda().float() for data in minibatches])
        all_t = torch.cat([data[3].cuda().float() for data in minibatches])

        all_y = torch.cat([data[7].cuda().float() for data in minibatches])

        all_attention = self.att_encoder(all_bh)

        src_mean = torch.mean(all_attention, dim=2)  # (288*1024)

        all_features = torch.cat((src_mean, all_freq, all_t), dim=1)

        # all_z = self.featurizer(all_features).squeeze(0)
        all_z = self.featurizer(all_features)

        # 领域鉴别器更新
        disc_input = all_z
        disc_input = Adver_network.ReverseLayerF.apply(
            disc_input, self.alpha)
        disc_out = self.domain_classifier(disc_input)
        disc_labels = torch.cat([
            torch.full((data[0].shape[0],), i,
                       dtype=torch.int64, device='cuda')
            for i, data in enumerate(minibatches)
        ])

        disc_out_s = disc_out.squeeze(0)
        # 这个应该不用改？
        # disc_loss = F.cross_entropy(disc_out, disc_labels)
        disc_loss = F.cross_entropy(disc_out_s, disc_labels)

        # 回归器更新
        all_preds = self.regressor(all_z).squeeze(0)

        # 回归器的loss需要更改为mse
        regressor_loss = F.mse_loss(all_preds, all_y)

        loss = regressor_loss + disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        if sch:
            sch.step()
        return {'total': loss.item(), 'class': regressor_loss.item(), 'dis': disc_loss.item()}

    def predict(self, all_bh, all_freq, all_t):
        all_attention = self.att_encoder(all_bh)
        # all_z = self.featurizer(all_bh, af, at).squeeze(0)
        src_mean = torch.mean(all_attention, dim=2)

        all_features = torch.cat((src_mean, all_freq, all_t), dim=1)

        # all_z = self.featurizer(all_features).squeeze(0)
        all_z = self.featurizer(all_features)

        return self.regressor(all_z)

