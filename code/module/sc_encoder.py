import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv

device = torch.device("cpu")
class Sc_encoder(nn.Module):
    def __init__(self, feats_dim,hidden_dim,feat_drop,attn_drop):
        super(Sc_encoder, self).__init__()
        in_size, layer_num_heads = feats_dim, 8
        self.gat_layers1 = GATConv(in_size, hidden_dim, layer_num_heads,
                                  feat_drop,attn_drop, activation=F.elu)
        self.layer44 = nn.Linear(hidden_dim * layer_num_heads,hidden_dim)

    def forward(self, x,A):
        x1 = self.gat_layers1(A,x).flatten(1)
        x1 = self.layer44(x1)
        return x1
