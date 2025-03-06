import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GCNLayer, GSRLayer, GraphConvolution  # existing layers
from preprocessing import normalize_adj_torch
from utils import get_device

device = get_device()

###########################################
#   Graph Encoder (and LowRes/HighRes)    #
###########################################

class GraphEncoder(nn.Module):
    def __init__(self, in_nodes, hidden_nodes, out_nodes, in_features, hidden_features, out_features):
        super(GraphEncoder, self).__init__()
        self.gc1 = GCNLayer(in_features, hidden_features, use_nonlinearity=True)
        self.gc2 = GCNLayer(hidden_features, hidden_features, use_nonlinearity=True)
        
        self.pool_linear1 = nn.Linear(hidden_features, hidden_nodes)
        
        self.gc3 = GCNLayer(hidden_features, out_features, use_nonlinearity=False)
        self.pool_linear2 = nn.Linear(out_features, out_nodes)
        
    def pool(self, x, adj, hidden=True):
        # x: (N, F), adj: (N, N)
        if hidden:
            # First pooling: map from N nodes to hidden_nodes.
            scores = self.pool_linear1(x)   # (N, hidden_nodes)
            # Softmax over nodes (dimension 0) so that assignments sum to 1.
            S = F.softmax(scores, dim=0)     # (N, hidden_nodes)
            S_t = S.transpose(0, 1)            # (hidden_nodes, N)
            x_latent = torch.mm(S_t, x)         # (hidden_nodes, F)
            adj_latent = torch.mm(torch.mm(S_t, adj), S)  # (hidden_nodes, hidden_nodes)
            adj_latent = 0.5 * (adj_latent + adj_latent.transpose(0, 1))
            return x_latent, adj_latent
        else:
            # Second pooling: map from current N to out_nodes.
            scores = self.pool_linear2(x)     # (N, out_nodes)
            S = F.softmax(scores, dim=0)        # (N, out_nodes)
            S_t = S.transpose(0, 1)             # (out_nodes, N)
            x_latent = torch.mm(S_t, x)          # (out_nodes, F)
            adj_latent = torch.mm(torch.mm(S_t, adj), S)  # (out_nodes, out_nodes)
            adj_latent = 0.5 * (adj_latent + adj_latent.transpose(0, 1))
            return x_latent, adj_latent

    def forward(self, x, adj):
        # x: (in_nodes, in_features), adj: (in_nodes, in_nodes)
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        x, adj = self.pool(x, adj, hidden=True)
        x = self.gc3(x, adj)
        x, adj = self.pool(x, adj, hidden=False)

        x = torch.clamp(x, 0, 1)
        adj = torch.clamp(adj, -1, 1)

        return x, adj

class LowResEncoder(GraphEncoder):
    def __init__(self, in_nodes, hidden_nodes, out_nodes, in_features, hidden_features, out_features):
        super(LowResEncoder, self).__init__(in_nodes, hidden_nodes, out_nodes, in_features, hidden_features, out_features)

class HighResEncoder(GraphEncoder):
    def __init__(self, in_nodes, hidden_nodes, out_nodes, in_features, hidden_features, out_features):
        super(HighResEncoder, self).__init__(in_nodes, hidden_nodes, out_nodes, in_features, hidden_features, out_features)

###########################################
#           Graph Decoder                 #
###########################################

class GraphDecoder(nn.Module):
    def __init__(self, in_nodes, hidden_nodes, out_nodes, in_features, hidden_features, out_features):
        super(GraphDecoder, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.pool = nn.Linear(in_nodes, out_nodes)

    def forward(self, x, A):
        # Transform features.
        h = F.relu(self.fc1(x))          # shape: (in_nodes, hidden_features)
        h_agg = torch.mm(A, h)             # shape: (in_nodes, hidden_features)
        h_trans = self.fc2(h_agg)          # shape: (in_nodes, out_features)
        # Pool nodes from in_nodes to out_nodes.
        h_trans_T = h_trans.transpose(0, 1)   # shape: (out_features, in_nodes)
        h_pooled_T = self.pool(h_trans_T)       # shape: (out_features, out_nodes)
        out_features = h_pooled_T.transpose(0, 1)  # shape: (out_nodes, out_features)
        # Reconstruct adjacency using the inner product.
        A_pred = torch.sigmoid(torch.mm(out_features, out_features.transpose(0, 1)))
        
        # Here is the clamping:
        # (Optionally, replace NaNs with a safe value.)
        A_pred = torch.nan_to_num(A_pred, nan=0.5, posinf=1.0, neginf=0.0)
        out_features = torch.clamp(out_features, -1, 1)
        A_pred = torch.clamp(A_pred, 0, 1)

        # print(A_pred)
        # print(out_features)
        
        return out_features, A_pred
