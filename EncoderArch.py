import torch 
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim, input_shape=(3, 128, 128)):
        super().__init__()
        self.input_shape = input_shape
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy = self.pool3(self.bn3(self.conv3(
                    self.pool2(self.bn2(self.conv2(
                        self.pool1(self.bn1(self.conv1(dummy))))
                    )))))
            self.flatten_dim = dummy.view(1, -1).shape[1]
        
        self.fc_mean = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
    
    def sample_latent_features(self, distribution):
        mean, logvar = distribution
        eps = torch.randn_like(logvar)
        return mean + torch.exp(0.5 * logvar) * eps
    
    def forward_conv(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        return x
    
    def forward(self, x):
        x = self.forward_conv(x)
        x = torch.flatten(x, start_dim=1)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        z = self.sample_latent_features((mean, logvar))
        return mean, logvar, z

"""
Example Testing:

encoder = Encoder(latent_dim=128)
sample_input = torch.randn(1, 3, 128, 128)

mean, logvar, latent_encoding = encoder(sample_input)
print("Mean Shape:", mean.shape)
print("Log Variance Shape:", logvar.shape)
print("Latent Encoding Shape:", latent_encoding.shape)
"""