import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, width).permute(0, 2, 1)  # BxWxC
        key = self.key_conv(x).view(batch_size, -1, width)  # BxCxW
        attention = self.softmax(torch.bmm(query, key))  # BxWxW
        value = self.value_conv(x).view(batch_size, -1, width)  # BxCxW

        out = torch.bmm(value, attention.permute(0, 2, 1))  # BxCxW
        out = self.gamma * out.view(batch_size, C, width) + x
        return out


class AudioEncoderAttention(nn.Module):
    def __init__(self, latent_dim=128, input_length=330750):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=10, stride=5, padding=0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4),
            SelfAttention1D(64),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2),
            SelfAttention1D(256)
        )

        self.flatten_dim = 256 * 64

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2)
        )

    def forward(self, x):
        assert x.shape[-1] == self.input_length

        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)

        mean_logvar = self.fc(x)
        mean, logvar = torch.chunk(mean_logvar, 2, dim=1)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

        return mean, logvar, z


class AudioDecoderAttention(nn.Module):
    def __init__(self, latent_dim=128, output_length=330750):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_length = output_length

        self.initial_length = 64
        self.initial_channels = 256

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.initial_channels * self.initial_length)
        )

        self.deconv_blocks = nn.Sequential(
            SelfAttention1D(256),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            SelfAttention1D(64),

            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(32, 16, kernel_size=10, stride=5, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(16, 8, kernel_size=12, stride=4, padding=4),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(8, 1, kernel_size=11, stride=11, padding=5)
        )

        self.length_adjust = (
            nn.ConstantPad1d((0, output_length - 330750), 0)
            if output_length > 330750 else None
        )

    def forward(self, z):
        x = self.fc(z).view(-1, self.initial_channels, self.initial_length)
        x = self.deconv_blocks(x)

        if self.length_adjust:
            x = self.length_adjust(x)

        x = x[:, :, :self.output_length]
        return torch.tanh(x)
    
'''from auido_wattention import AudioEncoderAttention, AudioDecoderAttention '''