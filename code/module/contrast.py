import torch
import torch.nn as nn


class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):#z1.shape=torch.Size([4019, 64])
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)#torch.Size([4019, 1])
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)#torch.Size([4019, 1])
        dot_numerator = torch.mm(z1, z2.t())#torch.Size([4019, 4019])
        dot_denominator = torch.mm(z1_norm, z2_norm.t())#torch.Size([4019, 4019])
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mp, z_sc, pos):
        z_proj_mp = self.proj(z_mp)#/gcn
        z_proj_sc = self.proj(z_sc)#/gcn
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)# torch.Size([4019, 4019])
        matrix_sc2mp = matrix_mp2sc.t()
        
        matrix_mp2sc = matrix_mp2sc/(torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)#torch.Size([4019, 4019])
        lori_mp = -torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()#各元素对应相乘

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()
        return self.lam * lori_mp + (1 - self.lam) * lori_sc
