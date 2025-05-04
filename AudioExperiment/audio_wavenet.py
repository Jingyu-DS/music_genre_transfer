import torch
import torch.nn as nn
import torch.nn.functional as F

### ---------------------- Helper Blocks ----------------------

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__(in_channels, out_channels, kernel_size,
                         padding=(kernel_size - 1) * dilation, dilation=dilation)

    def forward(self, x):
        out = super().forward(x)
        return out[:, :, :-self.padding[0]]  # Causal: remove right-side padding

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

### ---------------------- WaveNet Encoder ----------------------

class WaveNetEncoder(nn.Module):
    def __init__(self, latent_dim=32, input_length=90000):  # <- use input_length dynamically
        super(WaveNetEncoder, self).__init__()
        
        self.initial_conv = CausalConv1d(1, 32, kernel_size=3, dilation=1)

        self.wavenet_blocks = nn.ModuleList([
            WaveNetBlock(32, kernel_size=2, dilation=2**i) for i in range(6)
        ])
        
        self.downsample = nn.Conv1d(32, 64, kernel_size=4, stride=4)
        
        self.flatten = nn.Flatten()

        # Dynamically calculate flatten_dim
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)  # <- dynamic input length!
            dummy_out = self.forward_conv(dummy_input)
            flatten_dim = dummy_out.view(1, -1).shape[1]

        self.fc = nn.Linear(flatten_dim, latent_dim * 2)

    def forward_conv(self, x):
        x = self.initial_conv(x)
        skip_connections = 0
        for block in self.wavenet_blocks:
            x, skip = block(x)
            skip_connections = skip_connections + skip if isinstance(skip_connections, torch.Tensor) else skip
        x = skip_connections
        x = self.downsample(x)
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        x = self.flatten(x)
        x = self.fc(x)

        mu, logvar = torch.chunk(x, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return mu, logvar, z

### ---------------------- WaveNet Decoder ----------------------

class WaveNetDecoder(nn.Module):
    def __init__(self, latent_dim=32, output_length=90000):
        super(WaveNetDecoder, self).__init__()
        self.output_length = output_length
        
        self.fc = nn.Linear(latent_dim, 128 * (output_length // 256))

        self.upsample1 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=4)
        self.upsample2 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4)
        self.upsample3 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=4)

        self.wavenet_blocks = nn.ModuleList([
            WaveNetBlock(32, dilation=2**i) for i in range(6)
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
        out = F.interpolate(out, size=self.output_length, mode='linear', align_corners=False)
        return out

### ---------------------- Full CVAE Model ----------------------

class WaveNetCVAE(nn.Module):
    def __init__(self, latent_dim=32, input_length=90000):
        super(WaveNetCVAE, self).__init__()
        self.encoder = WaveNetEncoder(latent_dim=latent_dim, input_length=input_length)
        self.decoder = WaveNetDecoder(latent_dim=latent_dim, output_length=input_length)

    def forward(self, x):
        mu, logvar, z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

