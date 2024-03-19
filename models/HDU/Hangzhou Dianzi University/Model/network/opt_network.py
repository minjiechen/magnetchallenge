import torch
import torch.nn as nn
import torch.nn.functional as F

from network import Adver_network


class OptRegressor_0(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, num_domains=1):
        super(OptRegressor_0, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_domains),
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Seq2ERMpk(nn.Module):
    def __init__(self, seq2emb_module, projector_regressor):
        super(Seq2ERMpk, self).__init__()

        self.featurizer = seq2emb_module
        self.regressor = projector_regressor
        self.alpha = 1  # default 1

    def update(self, minibatches, opt, sch):
        all_bh = torch.cat([data[7].cuda().float() for data in minibatches])
        all_freq = torch.cat([data[2].cuda().float() for data in minibatches])
        all_t = torch.cat([data[3].cuda().float() for data in minibatches])
        all_pk = torch.cat([data[9].cuda().float() for data in minibatches])

        all_y = torch.cat([data[8].cuda().float() for data in minibatches])
        af = all_freq.unsqueeze(0)
        at = all_t.unsqueeze(0)
        apk = all_pk.unsqueeze(0)

        all_z = self.featurizer(all_bh, af, at, apk)
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

    def predict(self, bh, f, t, pk):
        return self.regressor(self.featurizer(bh, f, t, pk))



class Seq2Embedding_pk(nn.Module):
    # def __init__(self, sequence_encoder, projector_extractor, device):
    def __init__(self, sequence_encoder, projector_extractor):
        super().__init__()

        self.encoder = sequence_encoder
        self.features_extractor = projector_extractor

    def forward(self, source, var1, var2, var3):
        hidden, cell = self.encoder(source)
        concat_embedding = torch.cat((hidden, var1, var2, var3), 3)  # dim = 1 ?
        features = self.features_extractor(concat_embedding)
        return features