import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from utils.util import train_valid_target_eval_names, alg_loss_dict
from network import Adver_network
from network.attention_network import PositionalEncoder
import argparse
from datautil.getdataloader import get_LSTM_dataloader


class ViTBottleneck(nn.Module):
    def __init__(self,
                 input_size=2,
                 dim_val=32,
                 seq_size=1024 * 32,
                 patch_size=16,
                 embed_dim=256,
                 in_channel=1,
                 norm_layer=None,
                 distilled=None,
                 dropout_pos_enc=0.0,
                 max_seq_len=257,
                 n_heads=8,
                 dim_feedforward_encoder=1024,
                 dropout_encoder=0.0,
                 n_encoder_layers=2
                 ):
        super().__init__()

        self.input_size = input_size
        self.dim_val = dim_val
        self.seq_size = seq_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channel = in_channel
        self.norm_layer = norm_layer
        self.distilled = distilled
        self.dropout_pos_enc = dropout_pos_enc
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.dim_feedforward_encoder = dim_feedforward_encoder
        self.dropout_encoder = dropout_encoder
        self.n_encoder_layers = n_encoder_layers

        self.encoder_input_layer = nn.Sequential(nn.Linear(self.input_size, self.dim_val),
                                                 nn.Tanh(),
                                                 nn.Linear(self.dim_val, self.dim_val))
        # 288 * 1024 * 2 -> 288 * 1024 * 32
        self.pat_emd = PatchEmbeded(seq_size=self.seq_size, patch_size=self.patch_size, embed_dim=self.embed_dim,
                                    in_channel=self.in_channel, norm_layer=self.norm_layer,
                                    distilled=self.distilled)

        # 288 * 1024 * 32 -> 288 * 1 * 1024 * 32 -> 288 * 256 * 64 * 2 -> 288 * 256 * 128 -> 288 * 128 * 256

        self.pos_encoder = PositionalEncoder(d_model=self.embed_dim, dropout=self.dropout_pos_enc,
                                             max_len=self.max_seq_len)
        # 288 * 128 * 256
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.n_heads,
            dim_feedforward=self.dim_feedforward_encoder,
            dropout=self.dropout_encoder,
            activation="relu",
            batch_first=True
        )
        # 288 * 128 * 256

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                                         num_layers=self.n_encoder_layers,
                                                         norm=None)

        # 288 * 128 * 256

    def forward(self, x):
        src_input = self.encoder_input_layer(x)
        src_pat = self.pat_emd(src_input)
        src_pos = self.pos_encoder(src_pat)
        src_att = self.transformer_encoder(src_pos)

        return src_att


class PatchEmbeded(nn.Module):
    def __init__(self, seq_size=1024 * 32, patch_size=16, embed_dim=256, in_channel=1, norm_layer=None,
                 distilled=None):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channel = in_channel
        self.seq_size = seq_size
        self.num_tokens = 1
        self.drop_ratio = 0.0

        self.num_patches = self.seq_size // (self.patch_size * self.patch_size)

        self.proj = nn.Conv2d(self.in_channel, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) if distilled else None

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, self.embed_dim))

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.pos_drop = nn.Dropout(p=self.drop_ratio)

    def forward(self, x):
        # x: [B, C, H, W]
        # 288 * 1024 * 2 -> 288 * 1024 * 32 -> 288 * 1 * 1024 * 32 -> 288 * 256 * 64 * 2 ->
        # 288 * 256 * 128 -> 288 * 128 * 256
        x = x.unsqueeze(1)

        x = self.proj(x).flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]

        x = self.norm(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((x, cls_token), dim=1)  # [B, num_patches+1, embed_dim]
        # x = torch.cat((x, cls_token), dim=1)  # [B, num_patches+1, embed_dim] # 应该是cls token在前吧
        x = torch.cat((cls_token, x), dim=1)  # [B, num_patches+1, embed_dim]
        pos_embed = self.pos_embed

        out = self.pos_drop(x + pos_embed)  # [B, num_patches+1, embed_dim]

        return out


class Vit2DANN(nn.Module):
    # wip
    def __init__(self, vit_attention_module, projector_vit_head, projector_regressor, projector_discriminator):
        super(Vit2DANN, self).__init__()

        self.vit_encoder = vit_attention_module
        self.featurizer = projector_vit_head
        self.regressor = projector_regressor
        self.domain_classifier = projector_discriminator
        self.alpha = 1  # default 1

    def update(self, minibatches, opt, sch, DA_weight=0.1):
        all_bh = torch.cat([data[6].cuda().float() for data in minibatches])
        # all_freq = torch.cat([data[2].cuda().float() for data in minibatches])
        all_freq = torch.cat([data[9].cuda().float() for data in minibatches])
        all_t = torch.cat([data[3].cuda().float() for data in minibatches])

        all_y = torch.cat([data[7].cuda().float() for data in minibatches])

        all_attention = self.vit_encoder(all_bh)
        # all_z = self.featurizer(all_bh, af, at).squeeze(0)
        # cls_token = all_attention[:, 0]
        cls_token = all_attention[:, 0]
        # cls_token = all_attention[:, 0]
        # cls_token = all_attention[:, 0]

        all_features = torch.cat((cls_token, all_freq, all_t), dim=1)

        # all_z = self.featurizer(all_features).squeeze(0)
        all_z = self.featurizer(all_features)  # (288, 1)

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

        loss = regressor_loss + disc_loss * DA_weight
        opt.zero_grad()
        loss.backward()
        opt.step()

        if sch:
            sch.step()
        return {'total': loss.item(), 'class': regressor_loss.item(), 'dis': disc_loss.item()}

    def predict(self, x, y, z):
        all_attention = self.vit_encoder(x)
        # all_z = self.featurizer(all_bh, af, at).squeeze(0)
        cls_token = all_attention[:, 0]

        all_features = torch.cat((cls_token, y, z), dim=1)

        # all_z = self.featurizer(all_features).squeeze(0)
        all_z = self.featurizer(all_features)

        return self.regressor(all_z)


class Vit2ERM(nn.Module):
    # wip
    def __init__(self, vit_attention_module, projector_vit_head, projector_regressor):
        super(Vit2ERM, self).__init__()

        self.vit_encoder = vit_attention_module
        self.featurizer = projector_vit_head
        self.regressor = projector_regressor
        self.alpha = 1  # default 1

    def update(self, minibatches, opt, sch):
        all_bh = torch.cat([data[6].cuda().float() for data in minibatches])
        all_freq = torch.cat([data[9].cuda().float() for data in minibatches])
        all_t = torch.cat([data[3].cuda().float() for data in minibatches])
        all_y = torch.cat([data[7].cuda().float() for data in minibatches])

        all_attention = self.vit_encoder(all_bh)
        cls_token = all_attention[:, 0]

        all_features = torch.cat((cls_token, all_freq, all_t), dim=1)

        all_z = self.featurizer(all_features)

        # 回归器更新
        all_preds = self.regressor(all_z).squeeze(0)

        regressor_loss = F.mse_loss(all_preds, all_y)

        loss = regressor_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        if sch:
            sch.step()
        return {'total': loss.item()}

    def predict(self, seq, freq, t):
        all_attention = self.vit_encoder(seq)
        cls_token = all_attention[:, 0]
        all_features = torch.cat((cls_token, freq, t), dim=1)

        all_z = self.featurizer(all_features)

        return self.regressor(all_z)


class Vit2Norm(nn.Module):
    # wip
    def __init__(self, vit_attention_module, projector_vit_head, projector_regressor, projector_discriminator):
        super(Vit2Norm, self).__init__()

        self.vit_encoder = vit_attention_module
        self.featurizer = projector_vit_head
        self.regressor = projector_regressor
        self.domain_classifier = projector_discriminator
        self.alpha = 1  # default 1

    def update(self, minibatches, opt, sch, DA_weight=0.1):
        all_bh = torch.cat([data[6].cuda().float() for data in minibatches])
        # all_freq = torch.cat([data[2].cuda().float() for data in minibatches])
        all_freq = torch.cat([data[9].cuda().float() for data in minibatches])
        all_t = torch.cat([data[8].cuda().float() for data in minibatches])
        all_pk = torch.cat([data[7].cuda().float() for data in minibatches])

        all_y = torch.cat([data[10].cuda().float() for data in minibatches])

        all_attention = self.vit_encoder(all_bh)
        # all_z = self.featurizer(all_bh, af, at).squeeze(0)
        # cls_token = all_attention[:, 0]
        cls_token = all_attention[:, 0]
        # cls_token = all_attention[:, 0]
        # cls_token = all_attention[:, 0]

        all_features = torch.cat((cls_token, all_freq, all_t), dim=1)

        # all_z = self.featurizer(all_features).squeeze(0)
        all_z = self.featurizer(all_features)  # (288, 1)

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

        loss = regressor_loss + disc_loss * DA_weight
        opt.zero_grad()
        loss.backward()
        opt.step()

        if sch:
            sch.step()
        return {'total': loss.item(), 'class': regressor_loss.item(), 'dis': disc_loss.item()}

    def predict(self, x, y, z):
        all_attention = self.vit_encoder(x)
        # all_z = self.featurizer(all_bh, af, at).squeeze(0)
        cls_token = all_attention[:, 0]

        all_features = torch.cat((cls_token, y, z), dim=1)

        # all_z = self.featurizer(all_features).squeeze(0)
        all_z = self.featurizer(all_features)

        return self.regressor(all_z)
