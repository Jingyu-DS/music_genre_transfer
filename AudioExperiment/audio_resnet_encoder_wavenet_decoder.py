import torch
import torch.nn as nn
import torch.nn.functional as F

### ---------------------- Encoder ----------------------

class Resnet1DBlock(nn.Module):
    def __init__(self, filters, kernel_size, type='encode'):
        super(Resnet1DBlock, self).__init__()
        
        if type == 'encode':
            self.conv1a = nn.Conv1d(filters, filters, kernel_size, stride=2, padding=0)
            self.conv1b = nn.Conv1d(filters, filters, kernel_size, stride=1, padding=0)
            self.norm1a = nn.InstanceNorm1d(filters)
            self.norm1b = nn.InstanceNorm1d(filters)
            self.skip_conv = nn.Conv1d(filters, filters, 1, stride=2, padding=0)
        elif type == 'decode':
            self.conv1a = nn.ConvTranspose1d(filters, filters, kernel_size, stride=1, padding=0)
            self.conv1b = nn.ConvTranspose1d(filters, filters, kernel_size, stride=1, padding=0)
            self.norm1a = nn.BatchNorm1d(filters)
            self.norm1b = nn.BatchNorm1d(filters)
            self.skip_conv = None
        else:
            raise ValueError("Type must be 'encode' or 'decode'")
        
        self.type = type
    
    def forward(self, input_tensor):
        x = F.relu(input_tensor)
        x = self.conv1a(x)
        x = self.norm1a(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv1b(x)
        x = self.norm1b(x)
        x = F.leaky_relu(x, 0.4)
        residual = self.skip_conv(input_tensor) if self.type == 'encode' else input_tensor
        x += residual
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim=1024, input_length=90000):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        test_input = torch.randn(1, 1, 90000)
        test_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1, stride=2),
            nn.Conv1d(64, 64, kernel_size=1, stride=2),
            nn.Conv1d(64, 128, kernel_size=1, stride=2),
            nn.Conv1d(128, 128, kernel_size=1, stride=2),
            nn.Conv1d(128, 128, kernel_size=1, stride=2),
            nn.Conv1d(128, 128, kernel_size=1, stride=2),
            nn.Conv1d(128, 256, kernel_size=1, stride=2),
            nn.Conv1d(256, 256, kernel_size=1, stride=2),
        )
        with torch.no_grad():
            self.final_length = test_layers(test_input).shape[2]
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=1, stride=2)
        self.res1 = Resnet1DBlock(64, 1, type='encode')
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, stride=2)
        self.res2 = Resnet1DBlock(128, 1, type='encode')
        self.conv3 = nn.Conv1d(128, 128, kernel_size=1, stride=2)
        self.res3 = Resnet1DBlock(128, 1, type='encode')
        self.conv4 = nn.Conv1d(128, 256, kernel_size=1, stride=2)
        self.res4 = Resnet1DBlock(256, 1, type='encode')
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * self.final_length, latent_dim * 2)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.res2(x)
        x = self.conv3(x)
        x = self.res3(x)
        x = self.conv4(x)
        x = self.res4(x)
        x = self.flatten(x)
        x = self.fc(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z

### ---------------------- WaveNet Decoder ----------------------

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__(in_channels, out_channels, kernel_size,
                         padding=(kernel_size - 1) * dilation, dilation=dilation)
    
    def forward(self, x):
        out = super().forward(x)
        return out[:, :, :-self.padding[0]]


class WaveNetBlock(nn.Module):
    def __init__(self, channels, kernel_size=2, dilation=1):
        super().__init__()
        self.filter = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gate = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv_skip = nn.Conv1d(channels, channels, 1)
        self.conv_res = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        out = torch.tanh(self.filter(x)) * torch.sigmoid(self.gate(x))
        skip = self.conv_skip(out)
        res = self.conv_res(out)
        return x + res, skip


class Decoder(nn.Module):
    def __init__(self, latent_dim=1024, output_length=90000):
        super(Decoder, self).__init__()
        self.output_length = output_length
        self.fc = nn.Linear(latent_dim, 128 * (output_length // 256))

        self.upsample1 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=4)
        self.upsample2 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4)
        self.upsample3 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=4)

        self.wavenet_blocks = nn.ModuleList([
            WaveNetBlock(32, dilation=2 ** i) for i in range(6)
        ])

        self.final_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 128, -1)

        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)

        skip_connections = 0
        for block in self.wavenet_blocks:
            x, skip = block(x)
            skip_connections = skip_connections + skip if isinstance(skip_connections, torch.Tensor) else skip

        out = self.final_conv(skip_connections)
        out = F.interpolate(out, size=90000, mode='linear', align_corners=False)
        return out

### ---------------------- CVAE Model ----------------------

class CVAE(nn.Module):
    def __init__(self, latent_dim=1024):
        super(CVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, output_length=90000)

    def forward(self, x):
        mu, logvar, z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
