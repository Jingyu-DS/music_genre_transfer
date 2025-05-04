import torch
import torch.optim as optim
from Loss import get_loss

class TrainerCVAE:
    def __init__(self, trainloader, testloader, Encoder, Decoder, latent_dim, device="cuda"):
        # Set up device and transfer models to the device.
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder.to(self.device)
        self.decoder = Decoder.to(self.device)
        
        # Combine parameters of both encoder and decoder for joint optimization.
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=1e-4
        )
        
        # You can use a scheduler to adjust the learning rate based on validation loss.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Save the data loaders and latent dimensionality.
        self.train_loader = trainloader
        self.test_loader = testloader
        self.latent_dim = latent_dim

    def train(self, num_epochs=50, factor=100):
        for epoch in range(num_epochs):
            self.encoder.train()
            self.decoder.train()
            total_loss = 0

            # Training loop: iterate over batches
            for batch_idx, (x, condition) in enumerate(self.train_loader):
                # Ensure proper dimension ordering, if necessary.
                if x.dim() == 4 and x.shape[1] != 3:  # This reshapes if your channel dimension is last.
                    x = x.permute(0, 3, 1, 2)
                x = x.float().to(self.device)
                
                # Move conditioning information to the device.
                # Depending on your implementation the condition might be
                # a one-hot or continuous vector or an integer label.
                condition = condition.to(self.device)
                
                # Forward pass: pass both the input and the condition.
                mean, logvar, z = self.encoder(x, condition)
                x_reconstructed = self.decoder(z, condition)
                
                # Compute loss using your loss function; same formulation as VAE.
                loss_fn = get_loss(self.latent_dim, mean, logvar, factor, batch_size=x.shape[0])
                loss = loss_fn(x, x_reconstructed)
                
                # Backpropagation and update.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()

            # Validation loop over the test data.
            self.encoder.eval()
            self.decoder.eval()
            test_loss = 0
            with torch.no_grad():
                for x, condition in self.test_loader:
                    if x.dim() == 4 and x.shape[1] != 3:
                        x = x.permute(0, 3, 1, 2)
                    x = x.float().to(self.device)
                    condition = condition.to(self.device)
                    
                    mean, logvar, z = self.encoder(x, condition)
                    x_reconstructed = self.decoder(z, condition)
                    
                    loss_fn = get_loss(self.latent_dim, mean, logvar, factor, batch_size=x.shape[0])
                    test_loss += loss_fn(x, x_reconstructed).item()

            avg_train_loss = total_loss / len(self.train_loader)
            avg_test_loss = test_loss / len(self.test_loader)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

            self.scheduler.step(avg_test_loss)

        print("Training complete!")
        
