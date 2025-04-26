import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(channels, channels, kernel_size=2, dilation=dilation, padding=dilation)
        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.dilated_conv(x))
        out = self.conv_1x1(out)

        if out.size(2) > x.size(2):
            out = out[:, :, :x.size(2)]
        elif out.size(2) < x.size(2):
            x = x[:, :, :out.size(2)]

        return F.relu(out + x)


class WaveNetEncoder(nn.Module):
    def __init__(self, latent_dim=64, input_length=330750, channels=32, num_layers=5):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim

        self.initial_conv = nn.Conv1d(1, channels, kernel_size=1)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels, dilation=2 ** i) for i in range(num_layers)
        ])
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_mean = nn.Linear(channels, latent_dim)
        self.fc_logvar = nn.Linear(channels, latent_dim)

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.global_avg_pool(x).squeeze(-1)

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

        return mean, logvar, z


class WaveNetDecoder(nn.Module):
    def __init__(self, latent_dim=64, output_length=330750, channels=32, num_layers=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_length = output_length
        self.channels = channels
        self.initial_length = output_length // 256

        self.fc = nn.Linear(latent_dim, channels * self.initial_length)
        self.initial_conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=4)

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels, dilation=2 ** i) for i in range(num_layers)
        ])

        self.final_conv = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, z):
        x = self.fc(z).view(z.size(0), self.channels, self.initial_length)
        x = self.initial_conv(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.final_conv(x)

        if x.size(2) > self.output_length:
            x = x[:, :, :self.output_length]
        elif x.size(2) < self.output_length:
            pad_amount = self.output_length - x.size(2)
            x = F.pad(x, (0, pad_amount))

        return torch.tanh(x)
    
    '''from audio_wavenet import WaveNetEncoder, WaveNetDecoder '''