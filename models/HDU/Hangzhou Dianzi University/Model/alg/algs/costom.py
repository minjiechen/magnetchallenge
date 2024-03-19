# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm


class costom(Algorithm):
    def __init__(self, args):
        super(costom, self).__init__(args)

        # self.featurizer = get_fea(args)


        self.classifier = common_network.feat_classifier(
            class_num=args.num_classes, bottleneck_dim=512, type=args.classifier)

        self.network = self.classifier
        # self.loss_function = nn.MSELoss()
        self.loss_function = torch.nn.L1Loss()

    def update(self, minibatches, opt, sch):
        all_x = torch.cat([data[0].cuda().float() for data in minibatches])
        # print(all_x.ndim)
        all_y = torch.cat([data[1].cuda().float() for data in minibatches])
        # print(all_y)
        # loss = F.cross_entropy(self.predict(all_x), all_y)
        loss = self.loss_function(self.predict(all_x), all_y)

        # print(loss)
        # 原因是梯度爆炸了
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        # return {'class': loss.item()}
        return {'class': loss.item()}

    def predict(self, x):
        return self.network(x)

