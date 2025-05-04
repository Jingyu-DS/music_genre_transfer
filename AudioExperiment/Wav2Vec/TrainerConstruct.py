import torch
import torch.optim as optim
from Loss import get_loss


class GenreConditionedTrainer:
    def __init__(self, trainloader, testloader, Encoder, Decoder, lr, latent_dim, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder.to(self.device)
        self.decoder = Decoder.to(self.device)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=8,
            verbose=True
        )
        self.dataloader = trainloader
        self.testdataloader = testloader
        self.latent_dim = latent_dim

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def train(self, num_epochs=50, factor=5):

        for epoch in range(num_epochs):
            self.encoder.train()
            self.decoder.train()
            total_loss = 0

            for batch_idx, (x, genre_labels) in enumerate(self.dataloader):
                x = x.float().to(self.device)
                genre_labels = genre_labels.to(self.device)

                mean, logvar, z = self.encoder(x, genre_labels)
                x_reconstructed = self.decoder(z, genre_labels)

                loss_fn = get_loss(self.latent_dim, mean, logvar, factor, batch_size=x.shape[0])
                loss = loss_fn(x, x_reconstructed)

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()), max_norm=1.0
                )

                self.optimizer.step()

                total_loss += loss.item()

            self.encoder.eval()
            self.decoder.eval()
            test_loss = 0
            with torch.no_grad():
                for x, genre_labels in self.testdataloader:
                    x = x.float().to(self.device)
                    genre_labels = genre_labels.to(self.device)

                    mean, logvar, z = self.encoder(x, genre_labels)
                    x_reconstructed = self.decoder(z, genre_labels)
                    
                    loss_fn = get_loss(self.latent_dim, mean, logvar, factor, batch_size=x.shape[0])
                    test_loss += loss_fn(x, x_reconstructed).item()
            
            avg_loss = total_loss / len(self.dataloader)
            avg_test_loss = test_loss / len(self.testdataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        print("Training complete!")
