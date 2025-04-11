import torch 
import torch.nn as nn
import torch.nn.functional as F

class CVAE_Encoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, input_shape=(3, 128, 128), use_embedding=False, num_classes=None):
        super().__init__()
        self.input_shape = input_shape
        self.condition_dim = condition_dim
        self.use_embedding = use_embedding
        
        if use_embedding:
            if num_classes is None:
                raise ValueError("num_classes must be provided when using embedding for conditions.")
            self.embedding = nn.Embedding(num_classes, condition_dim)
        else:
            self.embedding = None
        
        # Convolutional layers to process the melspectrogram (or image) input.
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Compute the dimension of the flattened features after the conv layers.
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy = self.pool3(self.bn3(self.conv3(
                        self.pool2(self.bn2(self.conv2(
                            self.pool1(self.bn1(self.conv1(dummy)))
                        )))
                    )))
            self.flatten_dim = dummy.view(1, -1).shape[1]
        
        # Fully connected layers now take the flattened conv features concatenated with the condition vector.
        self.fc_mean = nn.Linear(self.flatten_dim + condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim + condition_dim, latent_dim)
    
    def sample_latent_features(self, mean, logvar):
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
    
    def forward(self, x, condition):
        # If using an embedding, convert condition labels to vectors.
        if self.use_embedding:
            # Assumes condition is of shape (batch,) with integer indices.
            condition = self.embedding(condition)  # Now shape is (batch, condition_dim)
        
        x = self.forward_conv(x)
        x = torch.flatten(x, start_dim=1)  # Shape: (batch, flatten_dim)
        
        # Concatenate the condition vector to the flattened convolutional features.
        x = torch.cat([x, condition], dim=1)  # Shape: (batch, flatten_dim + condition_dim)
        
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        z = self.sample_latent_features(mean, logvar)
        return mean, logvar, z

"""
# Example Testing:

# Example 1: Using one-hot or precomputed condition vectors (no embedding)
encoder_no_embed = CVAE_Encoder(latent_dim=128, condition_dim=10, input_shape=(3, 128, 128), use_embedding=False)
sample_input = torch.randn(1, 3, 128, 128)
sample_condition_vec = torch.randn(1, 10)  # For example, a one-hot encoded or continuous condition vector

mean, logvar, latent_encoding = encoder_no_embed(sample_input, sample_condition_vec)
print("Without embedding:")
print("Mean Shape:", mean.shape)            # Expected: (1, 128)
print("Log Var Shape:", logvar.shape)         # Expected: (1, 128)
print("Latent Encoding Shape:", latent_encoding.shape)  # Expected: (1, 128)

# Example 2: Using integer labels with an embedding layer.
encoder_with_embed = CVAE_Encoder(latent_dim=128, condition_dim=10, input_shape=(3, 128, 128),
                                   use_embedding=True, num_classes=5)
sample_condition_label = torch.tensor([2])  # Example condition label (e.g., representing genre index)
mean, logvar, latent_encoding = encoder_with_embed(sample_input, sample_condition_label)
print("\nWith embedding:")
print("Mean Shape:", mean.shape)            # Expected: (1, 128)
print("Log Var Shape:", logvar.shape)         # Expected: (1, 128)
print("Latent Encoding Shape:", latent_encoding.shape)  # Expected: (1, 128)
"""