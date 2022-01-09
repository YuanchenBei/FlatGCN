import torch
import numpy as np
import torch.nn as nn
import pickle as pkl
from os import path
import torch.nn.functional as F


class FlatGCN(nn.Module):
    def __init__(self, embs, layer_pairs, mlp_size=64, mlp_layer=3, if_xavier=True, drop_rate=0.5, device='cpu'):
        super(FlatGCN, self).__init__()
        self.X = embs
        self.mds = []
        for i in range(3):
            for j in range(3):
                self.mds.append((i, j))

        mlp_list = []
        self.u_bn = nn.ModuleList([
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device)
        ])
        self.i_bn = nn.ModuleList([
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device)
        ])
        self.lys = layer_pairs
        self.lys_bn = torch.nn.BatchNorm1d(len(self.lys), momentum=0.1).to(device)
        for i in range(mlp_layer - 1):
            if i == 0:
                pre_size = len(layer_pairs)
            else:
                pre_size = mlp_size
            linear = torch.nn.Linear(pre_size, mlp_size, bias=True).to(device)
            if if_xavier:
                nn.init.xavier_uniform_(linear.weight, gain=1)
            else:
                nn.init.kaiming_normal_(linear.weight)
            mlp_list.append(linear)
            mlp_list.extend([
                nn.BatchNorm1d(mlp_size, momentum=0.1).to(device),
                nn.LeakyReLU(),
                nn.Dropout(p=drop_rate)])
        if mlp_layer <= 1:
            pre_size = len(layer_pairs) * self.X.shape[-1]
        else:
            pre_size = mlp_size
        linear = torch.nn.Linear(pre_size, 1, bias=True).to(device)
        if if_xavier:
            torch.nn.init.xavier_uniform_(linear.weight)
        else:
            nn.init.kaiming_normal_(linear.weight)
        mlp_list.append(linear)
        self.mlp = torch.nn.Sequential(*mlp_list)
        if mlp_layer == 1:
            self.mlp[0].weight = torch.nn.Parameter(
                torch.FloatTensor(np.array(
                    [[1.0 / (len(layer_pairs))] * (len(layer_pairs))])).to(device))

    def forward(self, ids):
        xu = [self.u_bn[_](self.X[ids[:, 0]][:, _, :]) for _ in range(3)]
        xi = [self.i_bn[_](self.X[ids[:, 1]][:, _, :]) for _ in range(3)]
        p_list = [torch.sum(xu[ly[0]] * xi[ly[1]], dim=1, keepdim=True) for ly in self.lys]
        pred = self.lys_bn(torch.cat(p_list, dim=-1))
        pred_x = self.mlp(pred)
        return pred_x


class MLP(nn.Module):
    def __init__(self, embs, layer_pairs, mlp_size=64, mlp_layer=3, if_xavier=True, drop_rate=0.5, device='cpu'):
        super(MLP, self).__init__()
        self.X = embs
        mlp_list = [torch.nn.BatchNorm1d(6 * self.X.shape[-1], momentum=0.1).to(device)]
        for i in range(mlp_layer - 1):
            if i == 0:
                pre_size = 6 * self.X.shape[-1]
            else:
                pre_size = mlp_size

            linear = torch.nn.Linear(pre_size, mlp_size, bias=True).to(device)
            if if_xavier:
                nn.init.xavier_uniform_(linear.weight)
            mlp_list.append(linear)
            mlp_list.extend([
                nn.BatchNorm1d(mlp_size, momentum=0.1).to(device),
                nn.LeakyReLU(),
                nn.Dropout(p=drop_rate)])
        if mlp_layer <= 1:
            pre_size = 6 * self.X.shape[-1]
        else:
            pre_size = mlp_size
        linear = torch.nn.Linear(pre_size, 1, bias=True).to(device)
        if if_xavier:
            torch.nn.init.xavier_uniform_(linear.weight)
        mlp_list.append(linear)
        self.mlp = torch.nn.Sequential(*mlp_list)

    def forward(self, ids):
        batch = ids.shape[0]
        X_u = self.X[ids[:, 0]].reshape([batch, -1])
        X_i = self.X[ids[:, 1]].reshape([batch, -1])
        pred_x = self.mlp(torch.cat([X_u, X_i], dim=-1))
        return pred_x
