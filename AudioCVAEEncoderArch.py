import torch 
import torch.nn as nn
import torch.nn.functional as F

class AudioCVAEEncoder(nn.Module):
    def __init__(self, latent_dim=128, input_length=661500, condition_dim=10, use_embedding=False, num_classes=None):
        """
        Args:
            latent_dim (int): Size of the latent space.
            input_length (int): Length of the 1D audio input.
            condition_dim (int): Dimension of the condition vector (e.g. one-hot genre vector).
            use_embedding (bool): Whether to use an embedding layer for integer conditions.
            num_classes (int, optional): Number of classes.
        """
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.use_embedding = use_embedding
        
        if self.use_embedding:
            if num_classes is None:
                raise ValueError("num_classes must be provided when using embedding.")
            self.embedding = nn.Embedding(num_classes, condition_dim)
        else:
            self.embedding = None

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
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=0),  
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

        self.adaptive_pool = nn.AdaptiveAvgPool1d(128)
        self.initial_channels = 256
        self.flatten_dim = self.initial_channels * 128  # should be 32768
        
        # The fully connected layers now accept flattened conv features concatenated with the condition vector.
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim + condition_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2)  # Output both mean and logvar
        )

    def forward(self, x, condition):
        """
        Args:
            x (Tensor): Input audio waveform, shape (B, 1, input_length).
            condition (Tensor): Condition vector, shape (B, condition_dim), or integer labels if using embedding.
            
        Returns:
            mean (Tensor): Mean of the latent Gaussian.
            logvar (Tensor): Log variance of the latent Gaussian.
            z (Tensor): Sampled latent encoding.
        """
        # Verify input length.
        assert x.shape[-1] == self.input_length, \
            f"Input length {x.shape[-1]} doesn't match expected {self.input_length}"
            
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.view(x.size(0), -1)
        assert x.shape[1] == self.flatten_dim, \
            f"Flatten dimension mismatch: expected {self.flatten_dim}, got {x.shape[1]}"
        
        # If using embedding, assume condition contains integer labels.
        if self.use_embedding:
            condition = self.embedding(condition)  # Now condition is (B, condition_dim)
        # Otherwise, condition is assumed to already be a vector of shape (B, condition_dim)

        if condition.dim() == 1:
            condition = F.one_hot(condition, num_classes=self.condition_dim).float()
        
        # Concatenate flattened features and condition vector.
        x = torch.cat([x, condition], dim=1)
        
        mean_logvar = self.fc(x)
        mean, logvar = torch.chunk(mean_logvar, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        return mean, logvar, z

"""
# Testing the CVAE Encoder
if __name__ == "__main__":
    # Example usage:
    # Let's assume input_length=66150, latent_dim=128, and condition_dim=10 (10 genre classes).
    encoder = AudioCVAEEncoder(latent_dim=128, input_length=66150, condition_dim=10, use_embedding=False)
    sample_input = torch.randn(1, 1, 66150)  # (batch, channels, length)
    
    # Create a dummy condition vector (e.g., a one-hot encoded vector).
    # For instance, if the first genre is active, the one-hot vector might be [1, 0, 0, ..., 0]
    dummy_condition = torch.zeros(1, 10)
    dummy_condition[0, 0] = 1.0  # Assume genre index 0
    
    mean, logvar, latent_encoding = encoder(sample_input, dummy_condition)
    print("Mean Shape:", mean.shape)
    print("Log Variance Shape:", logvar.shape)
    print("Latent Encoding Shape:", latent_encoding.shape)
"""