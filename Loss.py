
import torch
import torch.nn.functional as F

def get_loss(latent_dim, distribution_mean, distribution_logvar, factor, batch_size):

    def get_reconstruction_loss(y_true, y_pred):
        """Computes MSE-based reconstruction loss"""
        reconstruction_loss = F.mse_loss(y_pred, y_true, reduction='sum') / batch_size
        return factor * reconstruction_loss  

    def get_kl_loss():
        """Computes KL divergence loss (KL(q||p))"""
        # More numerically stable calculation
        kl_loss = 1 + distribution_logvar - distribution_mean.pow(2) - distribution_logvar.exp()
        kl_loss = -0.5 * torch.sum(kl_loss, dim=1)  # Sum over latent dimensions
        return torch.mean(kl_loss)  # Average over batch

    def total_loss(y_true, y_pred):
        """Computes total VAE loss = reconstruction + KL"""
        return get_reconstruction_loss(y_true, y_pred) + get_kl_loss()

    return total_loss
