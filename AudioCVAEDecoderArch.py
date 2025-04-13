import torch 
import torch.nn as nn
import torch.nn.functional as F

class AudioCVAEDecoder(nn.Module):
    def __init__(self, latent_dim=128, condition_dim=10, output_length=661500):
        """
        Args:
            latent_dim (int): Dimension of the latent space.
            condition_dim (int): Dimension of the condition vector (e.g., one-hot genre vector).
            output_length (int): Length of the output audio waveform.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.output_length = output_length
        
        # Matching the encoder's compression settings.
        self.initial_length = 128
        self.initial_channels = 256 
        
        # The FC network now expects latent vector concatenated with condition vector.
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.initial_channels * self.initial_length)
        )
        
        # Transposed convolutions for upsampling.
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
        
        self.final_upsample = nn.ConvTranspose1d(8, 1, kernel_size=11, stride=11, padding=5)
        
        # Optionally adjust length if output_length exceeds the default (661500).
        self.length_adjust = nn.ConstantPad1d((0, output_length - 661500), 0) if output_length > 661500 else None

    def forward(self, z, condition):
        """
        Args:
            z (Tensor): Latent vector of shape (batch, latent_dim).
            condition (Tensor): Condition vector of shape (batch, condition_dim) (e.g., one-hot encoded genre).
        Returns:
            Tensor: Reconstructed audio waveform of shape (batch, 1, output_length).
        """
        # Concatenate latent vector and condition vector.
        x = torch.cat([z, condition], dim=1)
        # Project into a tensor suitable for the deconvolutional stack.
        x = self.fc(x)
        x = x.view(-1, self.initial_channels, self.initial_length)
        
        # Upsample using transposed convolutions.
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.final_upsample(x)
        
        # Adjust the length if needed.
        if self.length_adjust:
            x = self.length_adjust(x)
        x = x[:, :, :self.output_length]
        
        # Use tanh for output activation if the audio is normalized to [-1, 1].
        return torch.tanh(x)


"""
# Testing the CVAE Decoder
if __name__ == "__main__":
    decoder = AudioCVAE_Decoder(latent_dim=128, condition_dim=10, output_length=661500)
    latent_vector = torch.randn(1, 128)  # Example latent vector.
    
    # Create a dummy condition vector, e.g., one-hot for genre index 0.
    dummy_condition = torch.zeros(1, 10)
    dummy_condition[0, 0] = 1.0
    
    output_audio = decoder(latent_vector, dummy_condition)
    print(f"Decoder output shape: {output_audio.shape}")  # Expected: (1, 1, 661500)
"""