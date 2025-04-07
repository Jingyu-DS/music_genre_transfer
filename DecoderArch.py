import torch 
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.fc = nn.Linear(latent_dim, 32*16*16)
        
        # Transposed convolutions
        self.deconv1 = nn.ConvTranspose2d(32, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 32, 16, 16)  
        
        # Upsampling
        x = F.relu(self.bn1(self.deconv1(x)))  
        x = F.relu(self.bn2(self.deconv2(x)))  
        x = torch.sigmoid(self.deconv3(x))      
        
        return x

"""
Testing Decoder:

decoder = Decoder(latent_dim=LATENT_SPACE_SIZE)
latent_vector = torch.randn(1, LATENT_SPACE_SIZE)  
output_image = decoder(latent_vector)
print(f"Decoder output shape: {output_image.shape}")

"""