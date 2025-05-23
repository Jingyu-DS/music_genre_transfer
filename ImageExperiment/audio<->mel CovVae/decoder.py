class ImprovedDecoder(nn.Module):
    def __init__(self, latent_dim=512, out_channels=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 8 * 32),
            nn.BatchNorm1d(512 * 8 * 32),
            nn.ReLU(True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, out_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 512, 8, 32)
        x = self.deconv(h)
        return self.refine(x)
