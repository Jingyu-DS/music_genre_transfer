class ImprovedEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
        )
        self.flatten_dim = 512 * 8 * 32
        self.fc_mean = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mean, logvar)
        return mean, logvar, z
