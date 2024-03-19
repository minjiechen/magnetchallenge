import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from utils.util import train_valid_target_eval_names, alg_loss_dict
from network import Adver_network
import argparse
from datautil.getdataloader import get_LSTM_dataloader


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # self.lstm = nn.LSTM(1024, self.hidden_dim, num_layers=1, batch_first=True)
        self.lstm = nn.LSTM(2, self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.MLP = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, x):
        y = self.MLP(x)
        return y


# 特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self, num_var, hidden_dim, mod_dim):
        super(FeatureExtractor, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_var = num_var
        self.mod_dim = mod_dim

        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim + self.num_var, self.mod_dim),
            nn.Tanh(),
            nn.Linear(self.mod_dim, self.mod_dim),
            nn.Tanh(),
            nn.Linear(self.mod_dim, self.hidden_dim)
        )

    def forward(self, x):
        x = x.squeeze(0)
        y = self.out(x)
        y = y.unsqueeze(0)

        return y


# 回归器
class Regressor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, num_domains=1):
        super(Regressor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# 领域鉴别器
class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, num_domains=4):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# 将输入投影成embedding
class Seq2Embedding(nn.Module):
    # def __init__(self, sequence_encoder, projector_extractor, device):
    def __init__(self, sequence_encoder, projector_extractor):
        super().__init__()

        self.encoder = sequence_encoder
        self.features_extractor = projector_extractor

    def forward(self, source, var1, var2):
        hidden, cell = self.encoder(source)
        # concat_embedding = torch.cat((hidden, var1, var2), 2)  # dim = 1 ? 因为是在第三个维度连接，所以是2
        concat_embedding = torch.cat((hidden.squeeze(0), var1, var2), 1)  # 应该把hidden(1,batch,feat)出来的第0个维度squeeze掉
        # 这样输入var1和var2也不用进行unsqueeze了,不然后面bn层问题很大
        features = self.features_extractor(concat_embedding).squeeze(0)
        return features


class MLP2Embedding(nn.Module):
    # def __init__(self, sequence_encoder, projector_extractor, device):
    def __init__(self, sequence_encoder, projector_extractor):
        super().__init__()

        self.encoder = sequence_encoder
        self.features_extractor = projector_extractor

    def forward(self, source, var1, var2):
        feat_mlp = self.encoder(source)
        concat_embedding = torch.cat((feat_mlp, var1, var2), 1)
        features = self.features_extractor(concat_embedding).squeeze(0)
        return features


# 只负责把所有模块串联起来
class Seq2DANN(nn.Module):

    # def __init__(self, seq2emb_module, projector_regressor, projector_discriminator, device):
    def __init__(self, seq2emb_module, projector_regressor, projector_discriminator):
        super(Seq2DANN, self).__init__()

        self.featurizer = seq2emb_module
        self.regressor = projector_regressor
        self.domain_classifier = projector_discriminator
        # self.device = device
        # self.args = args
        self.alpha = 1  # default 1

    def update(self, minibatches, opt, sch):
        all_bh = torch.cat([data[6].cuda().float() for data in minibatches])
        # all_freq = torch.cat([data[2].cuda().float() for data in minibatches])
        all_freq = torch.cat([data[9].cuda().float() for data in minibatches])  # data[9] is log10 frequency
        all_t = torch.cat([data[3].cuda().float() for data in minibatches])

        all_y = torch.cat([data[7].cuda().float() for data in minibatches])
        # af = all_freq.unsqueeze(0)
        # at = all_t.unsqueeze(0)

        # all_z = self.featurizer(all_x)
        all_z = self.featurizer(all_bh, all_freq, all_t)
        # all_z = self.featurizer(all_bh, af, at).squeeze(0)

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
        # all_preds = self.regressor(all_z)

        # 回归器的loss需要更改
        # regressor_loss = F.cross_entropy(all_preds, all_y)
        regressor_loss = F.mse_loss(all_preds, all_y)
        # regressor_loss = F.cross_entropy(all_preds, all_y)

        loss = regressor_loss + disc_loss * 0.5
        opt.zero_grad()
        loss.backward()
        opt.step()

        if sch:
            sch.step()
        return {'total': loss.item(), 'class': regressor_loss.item(), 'dis': disc_loss.item()}

    def predict(self, x, y, z):
        return self.regressor(self.featurizer(x, y, z))

    def predict_transform(self, x, y, z):
        result = self.regressor(self.featurizer(x, y, z))

class Seq2ERM(nn.Module):
    def __init__(self, seq2emb_module, projector_regressor):
        super(Seq2ERM, self).__init__()

        self.featurizer = seq2emb_module
        self.regressor = projector_regressor
        self.alpha = 1  # default 1

    def update(self, minibatches, opt, sch):
        all_bh = torch.cat([data[6].cuda().float() for data in minibatches])
        all_freq = torch.cat([data[2].cuda().float() for data in minibatches])
        all_t = torch.cat([data[3].cuda().float() for data in minibatches])

        all_y = torch.cat([data[7].cuda().float() for data in minibatches])
        af = all_freq.unsqueeze(0)
        at = all_t.unsqueeze(0)

        all_z = self.featurizer(all_bh, af, at)
        # all_z = self.featurizer(all_bh, af, at).squeeze(0)

        # 回归器更新
        all_preds = self.regressor(all_z).squeeze(0)

        # 回归器的loss需要更改为mse
        regressor_loss = F.mse_loss(all_preds, all_y)

        loss = regressor_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        if sch:
            sch.step()
        return {'total': loss.item()}

    def predict(self, x, y, z):
        return self.regressor(self.featurizer(x, y, z))


class MLP2DANN(nn.Module):

    # def __init__(self, seq2emb_module, projector_regressor, projector_discriminator, device):
    def __init__(self, seq2emb_module, projector_regressor, projector_discriminator):
        super(MLP2DANN, self).__init__()

        self.featurizer = seq2emb_module
        self.regressor = projector_regressor
        self.domain_classifier = projector_discriminator
        # self.device = device
        # self.args = args
        self.alpha = 1  # default 1

    def update(self, minibatches, opt, sch):
        all_bh = torch.cat([data[6].cuda().float() for data in minibatches])
        # all_freq = torch.cat([data[2].cuda().float() for data in minibatches])
        all_freq = torch.cat([data[9].cuda().float() for data in minibatches])  # data[9] is log10 frequency
        all_t = torch.cat([data[3].cuda().float() for data in minibatches])

        all_y = torch.cat([data[7].cuda().float() for data in minibatches])
        abh = all_bh.reshape(all_bh.shape[0], -1)
        # af = all_freq.unsqueeze(0)
        # at = all_t.unsqueeze(0)

        # all_z = self.featurizer(all_x)
        all_z = self.featurizer(abh, all_freq, all_t)
        # all_z = self.featurizer(all_bh, af, at).squeeze(0)

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
        disc_loss = F.cross_entropy(disc_out_s, disc_labels)

        # 回归器更新
        all_preds = self.regressor(all_z).squeeze(0)

        regressor_loss = F.mse_loss(all_preds, all_y)

        loss = regressor_loss + disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        if sch:
            sch.step()
        return {'total': loss.item(), 'class': regressor_loss.item(), 'dis': disc_loss.item()}

    def predict(self, x, y, z):
        return self.regressor(self.featurizer(x, y, z))
