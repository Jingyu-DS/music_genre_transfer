
import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientLoss(nn.Module):
    def __init__(self, n_mels=128):
        super().__init__()
        self.register_buffer('freq_weights', torch.ones(1, 1, n_mels, 1))
        self.freq_weights[:, :, 30:80, :] = 1.3
    def forward(self, x, recon, mean, logvar, beta):
        weighted_mse = torch.mean(self.freq_weights * (x - recon)**2)
        x_grad_t = x[:, :, :, 1:] - x[:, :, :, :-1]
        recon_grad_t = recon[:, :, :, 1:] - recon[:, :, :, :-1]
        grad_loss = F.l1_loss(x_grad_t, recon_grad_t)
        recon_loss = 0.7 * weighted_mse + 0.3 * grad_loss
        mean_low, mean_high = torch.chunk(mean, 2, dim=1)
        logvar_low, logvar_high = torch.chunk(logvar, 2, dim=1)
        kl_low = -0.5 * torch.mean(1 + logvar_low - mean_low.pow(2) - logvar_low.exp())
        kl_high = -0.5 * torch.mean(1 + logvar_high - mean_high.pow(2) - logvar_high.exp())
        kl = kl_low + 0.5 * kl_high
        total_loss = recon_loss + beta * kl
        
        return total_loss, recon_loss, kl
