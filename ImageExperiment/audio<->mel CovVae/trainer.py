import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

class ImprovedTrainer:
    def __init__(self, trainloader, validloader, encoder, decoder, device="cpu"):
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.opt = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=5e-5, weight_decay=1e-5
        )
        self.scheduler = CosineAnnealingLR(self.opt, T_max=len(trainloader)*50, eta_min=1e-6)
        self.trainloader = trainloader
        self.validloader = validloader
        self.loss_history = {'train': [], 'val': [], 'recon': [], 'kl': []}
        os.makedirs("vae_logs", exist_ok=True)

    def loss_fn(self, x, recon, mean, logvar, beta):
        l1 = F.l1_loss(recon, x)
        l2 = F.mse_loss(recon, x)
        recon_loss = 0.7*l1 + 0.3*l2
        kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + beta*kl, recon_loss, kl

    def train(self, epochs=50, kl_warmup=15, max_beta=0.05):
        for ep in range(1, epochs+1):
            beta = min(max_beta, (ep/kl_warmup)**2 * max_beta)
            # training loop omitted for brevity
            # ...
            self.scheduler.step()
        # plotting omitted for brevity
