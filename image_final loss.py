
import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientLoss(nn.Module):
    def __init__(self, n_mels=128):
        super().__init__()
        w = torch.ones(1,1,n_mels,1)
        w[:,:,30:80,:] = 1.3
        self.register_buffer('freq_weights', w)

    def forward(self, x, recon, mean, logvar, beta):
        weighted_mse = torch.mean(self.freq_weights * (x-recon)**2)
        xg = x[:,:,:,1:] - x[:,:,:,:-1]
        rg = recon[:,:,:,1:] - recon[:,:,:,:-1]
        grad_loss = F.l1_loss(xg, rg)
        recon_loss = 0.7*weighted_mse + 0.3*grad_loss
        m_low,  m_high  = torch.chunk(mean,  2, dim=1)
        lv_low, lv_high = torch.chunk(logvar,2, dim=1)
        kl_low  = -0.5*torch.mean(1+lv_low - m_low.pow(2) - lv_low.exp())
        kl_high = -0.5*torch.mean(1+lv_high- m_high.pow(2)- lv_high.exp())
        kl = kl_low + 0.5*kl_high
        return recon_loss + beta*kl, recon_loss, kl
