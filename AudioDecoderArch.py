import torch 
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioDecoder(nn.Module):
    def __init__(self, latent_dim=128, output_length=661500):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_length = output_length
        
        # Matching the encoder's compression
        self.initial_length = 128  # Matches final length from encoder
        self.initial_channels = 256  # Matches final channels from encoder
        
        # Initial projection with residual connection
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.initial_channels * self.initial_length)
        )
        
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=3, padding=1), 
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=8, stride=4, padding=2),  
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2)
        )
        
        # Final reconstruction layers
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=10, stride=5, padding=3),  
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2)
        )
        
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=12, stride=4, padding=4),  
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2)
        )
        
        self.final_upsample = nn.ConvTranspose1d(8, 1, kernel_size=11, stride=11, padding=5)  # 61440->661500

        # Need to review this line as we might limit the audio to 15 - 25 seconds for this project
        self.length_adjust = nn.ConstantPad1d((0, output_length - 661500), 0) if output_length > 661500 else None

    def forward(self, z):
        # Project latent vector
        x = self.fc(z)
        x = x.view(-1, self.initial_channels, self.initial_length)
        
        # Upsampling stages
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        
        # Final output
        x = self.final_upsample(x)
        
        # Handle length adjustment if needed
        if self.length_adjust:
            x = self.length_adjust(x)
        
        # Ensure exact output length
        x = x[:, :, :self.output_length]
        
        return torch.tanh(x)

"""
Testing Decoder:

decoder = AudioDecoder(latent_dim=128, output_length=66150)
latent_vector = torch.randn(1, 128)  
output_audio = decoder(latent_vector)
print(f"Decoder output shape: {output_audio.shape}")  
"""