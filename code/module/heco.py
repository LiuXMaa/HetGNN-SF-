import torch.nn as nn
import torch.nn.functional as F
from .mp_encoder import Mp_encoder
from .sc_encoder import Sc_encoder
from .contrast import Contrast


class HeCo(nn.Module):
    def __init__(self, hidden_dim, feats_dim, feat_drop, attn_dropout, tau, lam):
        super(HeCo, self).__init__()
        self.hidden_dim = hidden_dim # 64
        self.fc_list = nn.Linear(feats_dim, feats_dim, bias=True)
        self.mp = Mp_encoder(feats_dim, hidden_dim,1, feat_drop)
        self.sc = Sc_encoder(feats_dim, hidden_dim, feat_drop, attn_dropout)
        self.contrast = Contrast(hidden_dim, tau, lam)

    def forward(self, x,adj,e,A,pos):
        x = self.fc_list(x)
        z_mp = self.mp(x,adj,e)
        z_sc = self.sc(x,A)
        loss = self.contrast(z_mp, z_sc, pos)
        return loss

    def get_embeds(self, x,adj,e,A):
        x = self.fc_list(x)
        # z_mp = self.mp(x,adj,e)
        # return z_mp.detach()
        z_sc = self.sc(x,A)
        return z_sc.detach()
