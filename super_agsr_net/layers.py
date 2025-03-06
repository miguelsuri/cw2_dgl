import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from preprocessing import normalize_adj_torch
from utils import get_device, weight_variable_glorot

device = get_device()


class GSRLayer(nn.Module):

    def __init__(self, lr_dim, hr_dim):
        super(GSRLayer, self).__init__()
        self.lr_dim = lr_dim
        self.hr_dim = hr_dim
        self.weights = torch.from_numpy(weight_variable_glorot(lr_dim * 2)).type(
            torch.FloatTensor
        )
        self.weights = torch.nn.Parameter(data=self.weights, requires_grad=True)

    def forward(self, A, X):
        lr = A
        lr_dim = lr.shape[0]
        f = X

        # print("-=-=-=-=-=-=-")

        eig_val_lr, U_lr = torch.linalg.eigh(lr, UPLO="U")
        # print(U_lr)
        eye_mat = torch.eye(lr_dim).type(torch.FloatTensor)
        s_d = torch.cat((eye_mat, eye_mat), 0).to(device)

        a = torch.matmul(self.weights, s_d)
        b = torch.matmul(a, torch.t(U_lr))

        f_d = torch.matmul(b, f)[: self.hr_dim, : self.hr_dim]
        # print(f_d)
        f_d = F.leaky_relu(f_d, negative_slope=0.2)
        # print(f_d)

        self.f_d = f_d.fill_diagonal_(1)
        # adj = normalize_adj_torch(self.f_d)
        adj = self.f_d
        # print(adj)

        X = torch.mm(adj, adj.t())
        X = (X + X.t()) / 2
        idx = torch.eye(self.hr_dim, dtype=bool)
        X[idx] = 1

        # print("-=-=-=-=-=-=-")

        return adj, torch.abs(X)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    # 160x320 320x320 =  160x320
    def __init__(self, in_features, out_features, dropout=0.0, act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        return output


class GCNLayer(nn.Module):
    """
    A single layer of a Graph Convolutional Network (GCN).
    ...
    """

    def __init__(self, input_dim, output_dim, use_nonlinearity=True):
        super(GCNLayer, self).__init__()
        self.use_nonlinearity = use_nonlinearity
        self.Omega = nn.Parameter(
            torch.randn(input_dim, output_dim)
            * torch.sqrt(torch.tensor(2.0) / (input_dim + output_dim))
        )
        self.beta = nn.Parameter(torch.zeros(output_dim))

    def forward(self, H_k, A_normalized):
        agg = torch.matmul(A_normalized, H_k)
        H_k_next = torch.matmul(agg, self.Omega) + self.beta
        return F.relu(H_k_next) if self.use_nonlinearity else H_k_next
