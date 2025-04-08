import torch
import torch.optim as optim
from Loss import get_loss

class Trainer:
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
        self.encoder.train()
        self.decoder.train()

        for epoch in range(num_epochs):
            total_loss = 0
            test_loss = 0
            
            for batch_idx, x in enumerate(self.dataloader):
                x = x.float().to(self.device)
                
                mean, logvar, z = self.encoder(x)
                x_reconstructed = self.decoder(z)

                loss_fn = get_loss(self.latent_dim, mean, logvar, factor, batch_size=x.shape[0])
                loss = loss_fn(x, x_reconstructed)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                for x in self.testdataloader:
                    x = x.float().to(self.device)
                    mean, logvar, z = self.encoder(x)
                    x_reconstructed = self.decoder(z)

                    loss_fn = get_loss(self.latent_dim, mean, logvar, factor, batch_size=x.shape[0])
                    test_loss += loss_fn(x, x_reconstructed).item()

            avg_loss = total_loss / len(self.dataloader)
            avg_test_loss = test_loss / len(self.testdataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        print("Training complete!")