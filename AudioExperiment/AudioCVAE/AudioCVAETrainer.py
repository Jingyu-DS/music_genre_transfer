import torch
import torch.optim as optim
from Loss import get_loss
import torch.nn.functional as F

class TrainerCVAE:
    def __init__(self, trainloader, testloader, Encoder, Decoder, latent_dim, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder.to(self.device)
        self.decoder = Decoder.to(self.device)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=1e-3
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        self.dataloader = trainloader
        self.testdataloader = testloader
        self.latent_dim = latent_dim

    def train(self, num_epochs=50, factor=100):
        for epoch in range(num_epochs):
            total_loss = 0
            test_loss = 0

            self.encoder.train()
            self.decoder.train()
            for batch_idx, (x, condition) in enumerate(self.dataloader):
                # Move both inputs and condition to the device.
                x = x.float().to(self.device)
                condition = condition.to(self.device)
                condition = F.one_hot(condition, num_classes=10).float()

                # Forward pass with condition
                mean, logvar, z = self.encoder(x, condition)
                x_reconstructed = self.decoder(z, condition)

                # Compute loss (reconstruction loss + KL divergence).
                loss_fn = get_loss(self.latent_dim, mean, logvar, factor, batch_size=x.shape[0])
                loss = loss_fn(x, x_reconstructed)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # Validation loop: switch models to evaluation mode.
            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                for x, condition in self.testdataloader:
                    x = x.float().to(self.device)
                    condition = condition.to(self.device)
                    condition = F.one_hot(condition, num_classes=10).float()

                    mean, logvar, z = self.encoder(x, condition)
                    x_reconstructed = self.decoder(z, condition)
                    loss_fn = get_loss(self.latent_dim, mean, logvar, factor, batch_size=x.shape[0])
                    test_loss += loss_fn(x, x_reconstructed).item()

            avg_loss = total_loss / len(self.dataloader)
            avg_test_loss = test_loss / len(self.testdataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

            self.scheduler.step(avg_test_loss)

        print("Training complete!")
