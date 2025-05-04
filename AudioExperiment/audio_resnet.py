import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        if self.type == 'encode':
            residual = self.skip_conv(input_tensor)
        else:
            residual = input_tensor
            
        x += residual
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim, input_length=90000):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Calculate final dimension after all downsampling
        test_input = torch.randn(1, 1, 90000)
        test_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1, stride=2, padding=0),
            nn.Conv1d(64, 64, kernel_size=1, stride=2, padding=0),
            nn.Conv1d(64, 128, kernel_size=1, stride=2, padding=0),
            nn.Conv1d(128, 128, kernel_size=1, stride=2, padding=0),
            nn.Conv1d(128, 128, kernel_size=1, stride=2, padding=0),
            nn.Conv1d(128, 128, kernel_size=1, stride=2, padding=0),
            nn.Conv1d(128, 256, kernel_size=1, stride=2, padding=0),
            nn.Conv1d(256, 256, kernel_size=1, stride=2, padding=0),
        )
        
        with torch.no_grad():
            self.final_length = test_layers(test_input).shape[2]
        
        # Define encoder layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=1, stride=2, padding=0)
        self.res1 = Resnet1DBlock(64, 1, type='encode')
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, stride=2, padding=0)
        self.res2 = Resnet1DBlock(128, 1, type='encode')
        self.conv3 = nn.Conv1d(128, 128, kernel_size=1, stride=2, padding=0)
        self.res3 = Resnet1DBlock(128, 1, type='encode')
        self.conv4 = nn.Conv1d(128, 256, kernel_size=1, stride=2, padding=0)
        self.res4 = Resnet1DBlock(256, 1, type='encode')
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * self.final_length, latent_dim * 2)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # x shape: [batch_size, 1, 90000]
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
        
        # Split into mu and logvar
        mu, logvar = torch.chunk(x, 2, dim=1)
        
        # Reparameterization
        z = self.reparameterize(mu, logvar)
        
        return mu, logvar, z


class Decoder(nn.Module):
    def __init__(self, latent_dim, final_length):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.final_length = final_length
        
        # Define decoder layers
        self.fc = nn.Linear(latent_dim, 512 * final_length)
        self.res1 = Resnet1DBlock(512, 1, type='decode')
        self.trans1 = nn.ConvTranspose1d(512, 512, kernel_size=1, stride=1, padding=0)
        self.res2 = Resnet1DBlock(512, 1, type='decode')
        self.trans2 = nn.ConvTranspose1d(512, 256, kernel_size=1, stride=1, padding=0)
        self.res3 = Resnet1DBlock(256, 1, type='decode')
        self.trans3 = nn.ConvTranspose1d(256, 128, kernel_size=1, stride=1, padding=0)
        self.res4 = Resnet1DBlock(128, 1, type='decode')
        self.trans4 = nn.ConvTranspose1d(128, 64, kernel_size=1, stride=1, padding=0)
        self.res5 = Resnet1DBlock(64, 1, type='decode')
        self.trans5 = nn.ConvTranspose1d(64, 1, kernel_size=1, stride=1, padding=0)
    
    def forward(self, z):
        # z shape: [batch_size, latent_dim]
        x = self.fc(z)
        x = x.view(x.size(0), 512, self.final_length)
        
        x = self.res1(x)
        x = self.trans1(x)
        x = self.res2(x)
        x = self.trans2(x)
        x = self.res3(x)
        x = self.trans3(x)
        x = self.res4(x)
        x = self.trans4(x)
        x = self.res5(x)
        x = self.trans5(x)
        
        # Upsample
        x = F.interpolate(x, size=90000, mode='linear', align_corners=False)
        
        return x


class CVAE(nn.Module):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, final_length=self.encoder.final_length)
    
    def forward(self, x):
        mu, logvar, z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar



