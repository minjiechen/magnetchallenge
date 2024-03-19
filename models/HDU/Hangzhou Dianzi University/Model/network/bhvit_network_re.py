import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from utils.util import train_valid_target_eval_names, alg_loss_dict
from network import Adver_network
from network.attention_network import PositionalEncoder
import argparse


def generate_square_subsequent_mask(sz1: int, sz2: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz1, sz2) * float('-inf'), diagonal=1)


class ViTBottleneck(nn.Module):
    def __init__(self,
                 input_size=1,
                 dim_val=32,
                 seq_size=1024 * 32,
                 patch_size=(8, 32),
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
        # 288 * 1024 * 1 -> 288 * 1024 * 32
        self.pat_emd = PatchEmbeded(seq_size=self.seq_size, patch_size=self.patch_size, embed_dim=self.embed_dim,
                                    in_channel=self.in_channel, norm_layer=self.norm_layer,
                                    distilled=self.distilled)

        # 288 * 1024 * 32 -> 288 * 1 * 1024 * 32 -> 288 * 256 * 64 * 2 -> 288 * 256 * 128 -> 288 * 128 * 256

        # self.pos_encoder = PositionalEncoder(d_model=self.embed_dim, dropout=self.dropout_pos_enc,
        #                                      max_len=self.max_seq_len)
        # 288 * 129 * 256     128channel + 1class_token
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
        # src_pos = self.pos_encoder(src_pat)  # patch_embedding中已包含可学习的位置编码
        src_att = self.transformer_encoder(src_pat)

        return src_att


class ViTDecoder(nn.Module):
    def __init__(self,
                 input_size=1,
                 dim_val=32,
                 seq_size=1024 * 32,
                 patch_size=16,
                 embed_dim=256,
                 in_channel=1,
                 norm_layer=None,
                 distilled=None,
                 dropout_pos_enc=0.0,
                 max_seq_len=257,
                 out_seq_len=129,
                 n_heads=8,
                 dim_feedforward_decoder=1024,
                 dropout_decoder=0.0,
                 n_decoder_layers=2
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
        self.out_seq_len = out_seq_len
        self.n_heads = n_heads
        self.dim_feedforward_decoder = dim_feedforward_decoder
        self.dropout_decoder = dropout_decoder
        self.n_decoder_layers = n_decoder_layers

        self.decoder_input_layer = nn.Sequential(
            nn.Linear(input_size, dim_val),
            nn.Tanh(),
            nn.Linear(dim_val, dim_val))

        self.pat_emd = PatchEmbeded(seq_size=self.seq_size, patch_size=self.patch_size, embed_dim=self.embed_dim,
                                    in_channel=self.in_channel, norm_layer=self.norm_layer,
                                    distilled=self.distilled)

        # self.pos_encoder = PositionalEncoder(d_model=self.embed_dim, dropout=self.dropout_pos_enc,
        #                                      max_len=self.max_seq_len)

        self.decoder_layer = nn.TransformerDecoderLayer(
            # d_model=self.dim_val,
            d_model=self.embed_dim,
            nhead=self.n_heads,
            dim_feedforward=self.dim_feedforward_decoder,
            dropout=self.dropout_decoder,
            activation="relu",
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=self.n_decoder_layers,
                                             norm=None)

        self.linear_mapping = nn.Sequential(
            nn.Linear(embed_dim, patch_size[0]),
            nn.Tanh(),
            nn.Linear(patch_size[0], patch_size[0])
        )

    def forward(self, src, tgt):
        tgt_input = self.decoder_input_layer(tgt)
        tgt_pat = self.pat_emd(tgt_input)[:, 1:, :]
        # tgt_pos = self.pos_encoder(tgt_pat)  # patch_embedding中已包含可学习的位置编码
        tgt_mask = generate_square_subsequent_mask(sz1=self.out_seq_len, sz2=self.out_seq_len).to(torch.device("cuda"))
        # tgt_mask = generate_square_subsequent_mask(sz1=self.out_seq_len, sz2=self.out_seq_len).to(torch.device("cuda"))
        # tgt_mask = generate_square_subsequent_mask(sz1=128, sz2=128).to(torch.device("cuda"))

        decode_result = self.decoder(
            tgt=tgt_pat,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=None
        )

        output = self.linear_mapping(decode_result)

        flatten_output = output.reshape(output.shape[0], -1)

        return flatten_output

        # (batch, )


class PatchEmbeded(nn.Module):
    def __init__(self, seq_size=1024 * 32, patch_size=(8, 32), embed_dim=256, in_channel=1, norm_layer=None,
                 # 16 * 16 = 256   8 * 32 = 256
                 distilled=None):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channel = in_channel
        self.seq_size = seq_size
        self.num_tokens = 1
        self.drop_ratio = 0.0

        self.num_patches = self.seq_size // (self.patch_size[0] * self.patch_size[1])

        self.proj = nn.Conv2d(self.in_channel, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) if distilled else None

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, self.embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))

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


class Vit2Norm(nn.Module):
    # wip
    def __init__(self, vit_attention_module, projector_vit_head, projector_regressor, projector_discriminator,
                 vit_decoder):
        super(Vit2Norm, self).__init__()

        self.vit_encoder = vit_attention_module
        self.featurizer = projector_vit_head
        self.regressor = projector_regressor
        self.domain_classifier = projector_discriminator
        self.vit_decoder = vit_decoder
        self.patch_len = 8
        self.alpha = 1  # default 1

    def update(self, minibatches, opt, sch, DA_weight=0.001):
        all_b = torch.cat([data[0].cuda().float() for data in minibatches])
        all_freq = torch.cat([data[1].cuda().float() for data in minibatches])
        all_t = torch.cat([data[2].cuda().float() for data in minibatches])
        all_pk = torch.cat([data[3].cuda().float() for data in minibatches])

        all_h = torch.cat([data[4].cuda().float() for data in minibatches])

        all_h_noize = torch.cat([data[5].cuda().float() for data in minibatches])

        # loss不用获取,domain在下面自动构造domain label矩阵

        all_kv = self.vit_encoder(all_b)

        # all_z = self.featurizer(all_bh, af, at).squeeze(0)
        # cls_token = all_attention[:, 0]

        all_features = torch.cat((all_kv, all_freq.repeat(1, all_kv.shape[1], 1),
                                  all_t.repeat(1, all_kv.shape[1], 1),
                                  all_pk.repeat(1, all_kv.shape[1], 1)),
                                 dim=2)

        # all_z = self.featurizer(all_features).squeeze(0)
        all_kv_p = self.featurizer(all_features)  # (288, 1)

        cls_token = all_kv_p[:, 0, :]

        # cls_token可能还是要的,用在领域鉴别器的分类上
        # 领域鉴别器更新
        disc_input = cls_token
        disc_input = Adver_network.ReverseLayerF.apply(
            disc_input, self.alpha)
        disc_out = self.domain_classifier(disc_input)
        disc_labels = torch.cat([
            torch.full((data[0].shape[0],), i,
                       dtype=torch.int64, device='cuda')
            for i, data in enumerate(minibatches)
        ])

        disc_loss = F.cross_entropy(disc_out, disc_labels)

        all_pred_h = self.vit_decoder(all_kv_p[:, 1:, :], all_h_noize)[:, :-8].unsqueeze(2)  # -8 is patch_len
        # 回归器更新

        # all_preds = self.regressor(all_z).squeeze(0)

        # 回归器的loss需要更改为mse
        regressor_loss = F.mse_loss(all_pred_h, all_h)

        loss = regressor_loss + disc_loss * DA_weight

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.25)
        opt.step()

        if sch:
            sch.step()
        return {'total': loss.item(), 'class': regressor_loss.item(), 'dis': disc_loss.item()}

    # predict还需要重写
    def predict(self, b, h_noize, freq, t, pk):
        kv = self.vit_encoder(b)
        # all_features = torch.cat(
        #     (all_kv, freq.unsqueeze(2).repeat(1, 129, 1), t.unsqueeze(2).repeat(1, 129, 1)), dim=2)

        all_features = torch.cat((kv, freq.repeat(1, kv.shape[1], 1),
                                  t.repeat(1, kv.shape[1], 1),
                                  pk.repeat(1, kv.shape[1], 1)),
                                 dim=2)

        kv_p = self.featurizer(all_features)  # (288, 1)

        pred_h = self.vit_decoder(kv_p[:, 1:, :], h_noize)[:, :-8].unsqueeze(2)

        return pred_h


class Vit2NormS(nn.Module):
    # wip
    def __init__(self, vit_attention_module, projector_vit_head, projector_regressor, vit_decoder):
        super(Vit2NormS, self).__init__()

        self.vit_encoder = vit_attention_module
        self.featurizer = projector_vit_head
        self.regressor = projector_regressor
        self.vit_decoder = vit_decoder
        self.patch_len = 8
        self.alpha = 1  # default 1

    def update(self, minibatches, opt, sch, DA_weight=0.001):
        all_b = torch.cat([data[0].cuda().float() for data in minibatches])
        all_freq = torch.cat([data[1].cuda().float() for data in minibatches])
        all_t = torch.cat([data[2].cuda().float() for data in minibatches])
        all_pk = torch.cat([data[3].cuda().float() for data in minibatches])

        all_h = torch.cat([data[4].cuda().float() for data in minibatches])

        all_h_noize = torch.cat([data[5].cuda().float() for data in minibatches])

        # loss不用获取,domain在下面自动构造domain label矩阵

        all_kv = self.vit_encoder(all_b)

        # all_z = self.featurizer(all_bh, af, at).squeeze(0)
        # cls_token = all_attention[:, 0]

        all_features = torch.cat((all_kv, all_freq.repeat(1, all_kv.shape[1], 1),
                                  all_t.repeat(1, all_kv.shape[1], 1),
                                  all_pk.repeat(1, all_kv.shape[1], 1)),
                                 dim=2)

        # all_z = self.featurizer(all_features).squeeze(0)
        all_kv_p = self.featurizer(all_features)  # (288, 1)

        cls_token = all_kv_p[:, 0, :]

        # cls_token可能还是要的,用在领域鉴别器的分类上
        # 领域鉴别器更新
        # disc_input = cls_token
        # disc_input = Adver_network.ReverseLayerF.apply(
        #     disc_input, self.alpha)
        # disc_out = self.domain_classifier(disc_input)
        # disc_labels = torch.cat([
        #     torch.full((data[0].shape[0],), i,
        #                dtype=torch.int64, device='cuda')
        #     for i, data in enumerate(minibatches)
        # ])

        # disc_loss = F.cross_entropy(disc_out, disc_labels)

        all_pred_h = self.vit_decoder(all_kv_p[:, 1:, :], all_h_noize)[:, :-8].unsqueeze(2)  # -8 is patch_len
        # 回归器更新

        # all_preds = self.regressor(all_z).squeeze(0)

        # 回归器的loss需要更改为mse
        regressor_loss = F.mse_loss(all_pred_h, all_h)

        # loss = regressor_loss + disc_loss * DA_weight
        loss = regressor_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.25)
        opt.step()

        if sch:
            sch.step()
        return {'class': regressor_loss.item()}

    # predict还需要重写
    def predict(self, b, h_noize, freq, t, pk):
        kv = self.vit_encoder(b)
        # all_features = torch.cat(
        #     (all_kv, freq.unsqueeze(2).repeat(1, 129, 1), t.unsqueeze(2).repeat(1, 129, 1)), dim=2)

        all_features = torch.cat((kv, freq.repeat(1, kv.shape[1], 1),
                                  t.repeat(1, kv.shape[1], 1),
                                  pk.repeat(1, kv.shape[1], 1)),
                                 dim=2)

        kv_p = self.featurizer(all_features)  # (288, 1)

        pred_h = self.vit_decoder(kv_p[:, 1:, :], h_noize)[:, :-8].unsqueeze(2)

        return pred_h
