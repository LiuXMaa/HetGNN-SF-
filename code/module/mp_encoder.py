import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class Mp_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, dropout):
        super(Mp_encoder, self).__init__()
        self.activation = F.elu

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GraphConv(input_dim, hidden_dim, activation=self.activation, weight=True))
        for i in range(layers - 1):
            self.gcn_layers.append(GraphConv(hidden_dim, hidden_dim, activation=self.activation, weight=True))
        self.gcn_layers.append(GraphConv(hidden_dim, hidden_dim, weight=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h, g,edge_weight):##如果不传e_weight参数，则默认为边权重为1
        for i, layers in enumerate(self.gcn_layers):
            h = self.dropout(h)
            h = layers(g, h,edge_weight=edge_weight)##如果不传e_weight参数，则默认为边权重为1
        return h
