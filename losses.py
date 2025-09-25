import torch
import torch.nn as nn
import torch.nn.functional as F


class ImputationLoss(nn.Module):
    def __init__(self, 
                 alpha_lmse, 
                 alpha_lcon, 
                 alpha_lconsis,
                 temperature=0.5, 
                 batch_size=32,
                 embed_dim=512,
                 device='cpu'):
        super().__init__()
        
        self.l_mse = nn.MSELoss()
        self.l_con = IntraModalityContrastLoss(batch_size, temperature, device)
        self.l_consis = ModalityConsisLoss(batch_size, temperature, embed_dim, device)
        
        self.alpha_lmse = alpha_lmse
        self.alpha_lcon = alpha_lcon
        self.alpha_lconsis = alpha_lconsis
        
    def forward(self, x_rec, y_rec, x_con, y_con):
        l_mse = self.alpha_lmse * self.l_mse(x_rec, y_rec)
        l_consis = self.alpha_lconsis * self.l_consis(x_con, y_con)
        l_con = self.alpha_lcon * self.l_con(x_con, y_con)
        return l_mse, l_consis, l_con
            
        
class IntraModalityContrastLoss(torch.nn.Module):
    def __init__(self, batch_size=1, temperature=0.5, device='cpu'):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(device))
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2 * 4, batch_size * 2 * 4, dtype=bool).to(device)).float())

    def forward(self, x_i, x_j):
        b, m, d = x_i.size()
        
        x_i = x_i.view(b * m, d)
        x_j = x_j.view(b * m, d)
        
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)

        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size * m)
        sim_ji = torch.diag(sim, -self.batch_size * m)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)

        denom = self.neg_mask * torch.exp(sim / self.temp)

        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size * m)


class ModalityConsisLoss(nn.Module):
    def __init__(self, batch_size=1, temperature=0.5, embed_dim=512, device='cpu'):
        super().__init__()
        self.projector = nn.Linear(embed_dim * 2, embed_dim).to(device)

        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(device))
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2 * 3, batch_size * 2 * 3, dtype=bool).to(device)).float())

    def _get_criterion(self, x_i, x_j)   :     
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size * 3)
        sim_ji = torch.diag(sim, -self.batch_size * 3)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size * 3)
    
    def forward(self, f_seq, f_spa):
        # input modality: [t1c, t1, t2, t2f]
        v_spa_t1c_t2 = self.projector(torch.cat([f_spa[:, 0], f_spa[:, 2]], dim=-1))
        v_seq_t1c_t2 = self.projector(torch.cat([f_seq[:, 0], f_seq[:, 2]], dim=-1))
        
        v_spa_t1_t2 = self.projector(torch.cat([f_spa[:, 1], f_spa[:, 2]], dim=-1))
        v_seq_t1_t2 = self.projector(torch.cat([f_seq[:, 1], f_seq[:, 2]], dim=-1))
        
        v_spa_t2_t2f = self.projector(torch.cat([f_spa[:, 3], f_spa[:, 2]], dim=-1))
        v_seq_t2_t2f = self.projector(torch.cat([f_seq[:, 3], f_seq[:, 2]], dim=-1))
        
        v_spa = torch.cat([v_spa_t1c_t2, v_spa_t1_t2, v_spa_t2_t2f], dim=0)
        v_seq = torch.cat([v_seq_t1c_t2, v_seq_t1_t2, v_seq_t2_t2f], dim=0)
        
        loss = self._get_criterion(v_spa, v_seq)
        
        return loss