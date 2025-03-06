import torch
import torch.nn as nn
from layers import GSRLayer, GraphConvolution
from ops import GraphDecoder, HighResEncoder, LowResEncoder
from preprocessing import normalize_adj_torch
import torch.nn.functional as F
import numpy as np
from utils import get_device


device = get_device()

class GSRNet(nn.Module):
    def __init__(self, args):
        super(GSRNet, self).__init__()
        self.lr_dim = args.lr_dim      # Low-resolution node count
        self.hr_dim = args.hr_dim      # High-resolution node count
        self.hidden_dim = args.hidden_dim
        self.embedding_size = args.embedding_size

        latent_nodes = self.lr_dim // 2  # Latent graph node count

        # Encoders: use identity matrices as node features.
        self.lr_encoder = LowResEncoder(
            in_nodes=self.lr_dim,
            hidden_nodes=latent_nodes,
            out_nodes=latent_nodes,
            in_features=self.lr_dim,   # One-hot features (identity)
            hidden_features=self.hidden_dim,
            out_features=self.embedding_size
        )
        
        self.hr_encoder = HighResEncoder(
            in_nodes=self.hr_dim,
            hidden_nodes=latent_nodes,  # Same latent count for consistency
            out_nodes=latent_nodes,
            in_features=self.hr_dim,      # One-hot features for HR graph
            hidden_features=self.hidden_dim,
            out_features=self.embedding_size
        )
        
        # IMPORTANT: The decoder now reconstructs a graph at the low-resolution dimension!
        self.decoder = GraphDecoder(
            in_nodes=latent_nodes,
            hidden_nodes=None,
            out_nodes=self.lr_dim,      # <-- Set to lr_dim (not hr_dim)
            in_features=self.embedding_size,
            hidden_features=self.hidden_dim,
            out_features=self.hidden_dim
        )
        
        # The GSR layer upsamples from the LR dimension to HR dimension.
        self.layer = GSRLayer(self.lr_dim, self.hr_dim)
        self.gc1 = GraphConvolution(self.hr_dim, self.hidden_dim, 0, act=F.relu)
        self.gc2 = GraphConvolution(self.hidden_dim, self.hr_dim, 0, act=F.relu)
    
    def forward(self, lr, hr=None):
        # Create identity feature matrices for LR and (if available) HR.
        I_lr = torch.eye(self.lr_dim).to(device)
        A_lr = normalize_adj_torch(lr).to(device)
        latent_lr, latent_adj_lr = self.lr_encoder(I_lr, A_lr)

        if hr is not None:
            I_hr = torch.eye(self.hr_dim).to(device)
            A_hr = normalize_adj_torch(hr).to(device)
            latent_hr, latent_adj_hr = self.hr_encoder(I_hr, A_hr)
        else:
            latent_hr, latent_adj_hr = None, None
        



        # Decoder reconstructs a low-resolution graph.
        decoded_features, decoded_adj = self.decoder(latent_lr, latent_adj_lr)
        
        # GSRLayer upsamples the decoded (lr) graph to high-resolution.
        # self.net_outs = decoded_features
        # print("-=-=-=-=-=-=-")
        # print(self.net_outs)
        # print(decoded_adj)

        self.outputs, self.Z = self.layer(decoded_adj, decoded_features)
        # print(self.outputs)
        # print(self.Z)
        self.hidden1 = self.gc1(self.Z, self.outputs)
        # print(self.hidden1)
        self.hidden2 = self.gc2(self.hidden1, self.outputs)
    

        z = self.hidden2
        # print(z)
        z = (z + z.transpose(0, 1)) / 2
        # print(z)
        idx = torch.eye(self.hr_dim, dtype=torch.bool).to(device)
        # print(z)
        z[idx] = 1
        # print(z)
        # print("-=-=-=-=-=-=-")
        
        return torch.relu(z), latent_lr, latent_hr, decoded_features, decoded_adj



class Dense(nn.Module):
    def __init__(self, n1, n2, args):
        super(Dense, self).__init__()
        self.weights = torch.nn.Parameter(torch.FloatTensor(n1, n2), requires_grad=True)
        nn.init.normal_(self.weights, mean=args.mean_dense, std=args.std_dense)

    def forward(self, x):
        out = torch.mm(x, self.weights)
        return out


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.dense_1 = Dense(args.hr_dim, args.hr_dim, args)
        self.relu_1 = nn.ReLU()
        self.dense_2 = Dense(args.hr_dim, args.hr_dim, args)
        self.relu_2 = nn.ReLU()
        self.dense_3 = Dense(args.hr_dim, 1, args)
        self.sigmoid = nn.Sigmoid()
        self.dropout_rate = args.dropout_rate

    def forward(self, x):
        x = F.dropout(self.relu_1(self.dense_1(x)), self.dropout_rate) + x
        x = F.dropout(self.relu_2(self.dense_2(x)), self.dropout_rate) + x
        x = self.dense_3(x)
        epsilon = 1e-6
        return torch.clamp(x, epsilon, 1 - epsilon)


def gaussian_noise_layer(input_layer, args):
    z = torch.empty_like(input_layer)
    noise = z.normal_(mean=args.mean_gaussian, std=args.std_gaussian)
    z = torch.abs(input_layer + noise)

    z = (z + z.t()) / 2
    z = z.fill_diagonal_(1)
    return z
