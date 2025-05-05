import torch
import torch.nn as nn

class LightResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class MemoryEfficientEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            LightResBlock(64),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            LightResBlock(128),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            LightResBlock(256),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )
        self.flatten_dim = 512 * 4 * 16
        self.fc_mean_low  = nn.Linear(self.flatten_dim, latent_dim//2)
        self.fc_logvar_low= nn.Linear(self.flatten_dim, latent_dim//2)
        self.fc_mean_high = nn.Linear(self.flatten_dim, latent_dim//2)
        self.fc_logvar_high=nn.Linear(self.flatten_dim, latent_dim//2)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        h = self.conv_layers(x)
        h_flat = h.view(h.size(0), -1)
        m_low = self.fc_mean_low(h_flat);  lv_low = self.fc_logvar_low(h_flat);  z_low = self.reparameterize(m_low, lv_low)
        m_high= self.fc_mean_high(h_flat); lv_high= self.fc_logvar_high(h_flat); z_high= self.reparameterize(m_high, lv_high)
        mean   = torch.cat([m_low,  m_high],  dim=1)
        logvar = torch.cat([lv_low, lv_high], dim=1)
        z      = torch.cat([z_low,  z_high],  dim=1)
        return mean, logvar, z
