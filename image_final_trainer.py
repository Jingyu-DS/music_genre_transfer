import os, gc, numpy as np, torch
from image_final_loss import EfficientLoss

class MemoryEfficientTrainer:
    def __init__(self, trainloader, validloader, encoder, decoder, device="cpu"):
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        
        self.opt = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=1e-4,
            weight_decay=1e-6,
            betas=(0.9, 0.999)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=len(trainloader) * 5, eta_min=1e-5
        )
        # loss
        self.loss_fn = EfficientLoss(n_mels=128).to(self.device)
        self.trainloader = trainloader
        self.validloader = validloader
        self.loss_history = {'train': [], 'val': [], 'recon': [], 'kl': []}
        os.makedirs("vae_logs", exist_ok=True)
    
    def train(self, epochs=50, warmup=5, max_beta=0.01, min_beta=0.001, 
              accumulation_steps=4): 
        for epoch in range(1, epochs+1):
            cycle_position = (epoch % 10) / 10
            beta = min_beta + (max_beta - min_beta) * min(1.0, epoch / (epochs * 0.7))
            beta = beta * (0.8 + 0.2 * np.sin(cycle_position * 2 * np.pi))
            self.encoder.train(); self.decoder.train()
            train_loss = 0
            train_recon_loss = 0
            train_kl_loss = 0
            
            self.opt.zero_grad()
            
            for batch_idx, (x, _) in enumerate(self.trainloader):
                x = x.float().to(self.device)
                x = 2.0 * ((x - x.min()) / (x.max() - x.min() + 1e-8)) - 1.0
                mean, logvar, z = self.encoder(x)
                recon = self.decoder(z)
                loss, recon_loss, kl_loss = self.loss_fn(x, recon, mean, logvar, beta)

                loss = loss / accumulation_steps
                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(self.trainloader):
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.decoder.parameters()),
                        max_norm=1.0
                    )

                    self.opt.step()
                    self.scheduler.step()
                    self.opt.zero_grad()

                train_loss += loss.item() * accumulation_steps
                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
                
                if batch_idx % 5 == 0:
                    del mean, logvar, z, recon
                    torch.cuda.empty_cache()
                    gc.collect()
                

                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}/{epochs}, Batch {batch_idx}/{len(self.trainloader)}, "
                          f"Beta: {beta:.4f}, Loss: {loss.item() * accumulation_steps:.4f}")
            

            avg_train = train_loss / len(self.trainloader)
            avg_recon = train_recon_loss / len(self.trainloader)
            avg_kl = train_kl_loss / len(self.trainloader)


            self.encoder.eval(); self.decoder.eval()
            val_loss = 0
            
            with torch.no_grad():
                for x, _ in self.validloader:
                    x = x.float().to(self.device)
                    x = 2.0 * ((x - x.min()) / (x.max() - x.min() + 1e-8)) - 1.0
                    
                    mean, logvar, z = self.encoder(x)
                    recon = self.decoder(z)
                    loss, _, _ = self.loss_fn(x, recon, mean, logvar, beta)
                    
                    val_loss += loss.item()

                    del mean, logvar, z, recon, x
                    torch.cuda.empty_cache()
                    gc.collect()
            
            avg_val = val_loss / len(self.validloader)
            self.loss_history['train'].append(avg_train)
            self.loss_history['val'].append(avg_val)
            self.loss_history['recon'].append(avg_recon)
            self.loss_history['kl'].append(avg_kl)
            

            if epoch % 10 == 0 or epoch == epochs:
                model_path = f"vae_logs/audio_vae_epoch_{epoch}.pt"
                torch.save({
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                    'epoch': epoch,
                }, model_path)
            
            print(f"Epoch {epoch}/{epochs}  β={beta:.4f}  Train={avg_train:.4f}  Recon={avg_recon:.4f}  KL={avg_kl:.4f}  Val={avg_val:.4f}")
            

            torch.cuda.empty_cache()
            gc.collect()


        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.loss_history['train'], label='Train Loss')
        plt.plot(self.loss_history['val'], label='Val Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.legend(); plt.grid(True)
        plt.title('Total Loss')
        
        plt.subplot(1, 3, 2)
        plt.plot(self.loss_history['recon'], label='Recon Loss')
        plt.plot(self.loss_history['kl'], label='KL Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.legend(); plt.grid(True)
        plt.title('Loss Components')
        
        plt.subplot(1, 3, 3)

        betas = []
        for e in range(1, epochs+1):
            cycle_pos = (e % 10) / 10
            b = min_beta + (max_beta - min_beta) * min(1.0, e / (epochs * 0.7))
            b = b * (0.8 + 0.2 * np.sin(cycle_pos * 2 * np.pi))
            betas.append(b)
        plt.plot(range(1, epochs+1), betas)
        plt.xlabel('Epoch'); plt.ylabel('Beta')
        plt.title('KL Weight (Beta)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('vae_logs/training_loss.png')
        plt.close()
        

        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'epoch': epochs,
        }, "vae_logs/audio_vae_final.pt")
