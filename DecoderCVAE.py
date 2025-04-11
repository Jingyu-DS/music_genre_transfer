import torch 
import torch.nn as nn
import torch.nn.functional as F

class CVAE_Decoder(nn.Module):
    def __init__(self, latent_dim, condition_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        # Now the FC layer must accept the latent vector concatenated with the condition vector.
        self.fc = nn.Linear(latent_dim + condition_dim, 32 * 16 * 16)
        
        # Transposed convolution layers to upsample
        self.deconv1 = nn.ConvTranspose2d(32, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, z, condition):
        # Concatenate latent vector with the condition vector along the feature dimension.
        x = torch.cat([z, condition], dim=1)
        
        # Project into a feature map and reshape accordingly
        x = self.fc(x)
        x = x.view(-1, 32, 16, 16)
        
        # Apply transposed convolutions (with batch normalization and ReLU activations)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = torch.sigmoid(self.deconv3(x))
        
        return x


"""
# Example Testing:

# Replace these with your actual latent and condition dimensions.
LATENT_SPACE_SIZE = 128
CONDITION_DIM = 10

# Create an instance of the CVAE decoder.
decoder = CVAE_Decoder(latent_dim=LATENT_SPACE_SIZE, condition_dim=CONDITION_DIM)

# Create dummy inputs: latent vector and a condition vector (which might be a one-hot genre representation).
latent_vector = torch.randn(1, LATENT_SPACE_SIZE)
condition_vector = torch.randn(1, CONDITION_DIM)  # Alternatively, use a one-hot vector of shape (1, 10)

output = decoder(latent_vector, condition_vector)
print(f"Decoder output shape: {output.shape}")
"""