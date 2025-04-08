import torch 
import torch.nn as nn
import torch.nn.functional as F

class AudioEncoder(nn.Module):
    def __init__(self, latent_dim=128, input_length=661500):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=10, stride=5, padding=0),  
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4, 4)  
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=0),  
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4, 4)  
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=0),  # 2066 -> 1031
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2, 2)  
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=0),  
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2, 2)  
        )

        self.flatten_dim = 256 * 128

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim*2)  # Output both mean and logvar
        )

    def forward(self, x):
        # Verify input length
        assert x.shape[-1] == self.input_length, \
            f"Input length {x.shape[-1]} doesn't match expected {self.input_length}"

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.view(x.size(0), -1)
        assert x.shape[1] == self.flatten_dim, \
            f"Flatten dimension mismatch: expected {self.flatten_dim}, got {x.shape[1]}"
        
        # Split into mean and logvar
        mean_logvar = self.fc(x)
        mean, logvar = torch.chunk(mean_logvar, 2, dim=1)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        
        return mean, logvar, z
"""
encoder = AudioEncoder(latent_dim=128, input_length=66150) 
sample_input = torch.randn(1, 1, 66150)  # (batch, channels, length)
    
mean, logvar, latent_encoding = encoder(sample_input)
print("Mean Shape:", mean.shape)
print("Log Variance Shape:", logvar.shape)
print("Latent Encoding Shape:", latent_encoding.shape)

"""