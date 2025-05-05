import os, gc, numpy as np, torch
from loss import EfficientLoss


class MemoryEfficientTrainer:
    def __init__(self, trainloader, validloader, encoder, decoder, device="cpu"):
        self.device   = torch.device(device)
        self.encoder  = encoder.to(self.device)
        self.decoder  = decoder.to(self.device)
        self.opt      = torch.optim.AdamW(
            list(encoder.parameters())+list(decoder.parameters()),
            lr=1e-4, weight_decay=1e-6, betas=(0.9,0.999))
        self.scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=len(trainloader)*5, eta_min=1e-5)
        self.loss_fn  = EfficientLoss().to(self.device)
        self.trainloader, self.validloader = trainloader, validloader
        self.loss_history = {'train':[],'val':[],'recon':[],'kl':[]}
        os.makedirs("vae_logs", exist_ok=True)

    def train(self, epochs=50, warmup=5, max_beta=0.01, min_beta=0.001, accumulation_steps=4):
        for epoch in range(1,epochs+1):
            cycle_pos = (epoch%10)/10
            beta = min_beta + (max_beta-min_beta)*min(1.0,epoch/(epochs*0.7))
            beta *= (0.8 + 0.2*np.sin(cycle_pos*2*np.pi))
            self.encoder.train(); self.decoder.train()
            tl, trl, tkl = 0,0,0
            self.opt.zero_grad()
            for i,(x,_) in enumerate(self.trainloader):
                x = x.float().to(self.device)
                x = 2*((x-x.min())/(x.max()-x.min()+1e-8))-1
                mean, logvar, z = self.encoder(x)
                recon = self.decoder(z)
                loss, rl, kl = self.loss_fn(x,recon,mean,logvar,beta)
                loss = loss/accumulation_steps
                loss.backward()
                if (i+1)%accumulation_steps==0 or (i+1)==len(self.trainloader):
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters())+list(self.decoder.parameters()),1.0)
                    self.opt.step(); self.scheduler.step(); self.opt.zero_grad()
                tl  += loss.item()*accumulation_steps
                trl += rl.item()
                tkl += kl.item()
                if i%5==0:
                    del mean,logvar,z,recon; torch.cuda.empty_cache(); gc.collect()
            avg_train = tl/len(self.trainloader)
            avg_recon = trl/len(self.trainloader)
            avg_kl    = tkl/len(self.trainloader)
            val_loss=0
            self.encoder.eval(); self.decoder.eval()
            with torch.no_grad():
                for x,_ in self.validloader:
                    x = x.float().to(self.device)
                    x = 2*((x-x.min())/(x.max()-x.min()+1e-8))-1
                    mean, logvar, z = self.encoder(x)
                    recon = self.decoder(z)
                    loss, _, _ = self.loss_fn(x,recon,mean,logvar,beta)
                    val_loss += loss.item()
                    del mean,logvar,z,recon,x; torch.cuda.empty_cache(); gc.collect()
            avg_val = val_loss/len(self.validloader)
            self.loss_history['train'].append(avg_train)
            self.loss_history['val'].append(avg_val)
            self.loss_history['recon'].append(avg_recon)
            self.loss_history['kl'].append(avg_kl)
            if epoch%10==0 or epoch==epochs:
                torch.save({'encoder':self.encoder.state_dict(),
                            'decoder':self.decoder.state_dict(),
                            'epoch':epoch},
                           f"vae_logs/audio_vae_epoch_{epoch}.pt")
            print(f"Epoch {epoch}/{epochs} β={beta:.4f} Train={avg_train:.4f} Recon={avg_recon:.4f} KL={avg_kl:.4f} Val={avg_val:.4f}")

        import matplotlib.pyplot as plt
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1); plt.plot(self.loss_history['train'],label='Train'); plt.plot(self.loss_history['val'],label='Val'); plt.legend(); plt.grid(); plt.title('Total')
        plt.subplot(1,3,2); plt.plot(self.loss_history['recon'],label='Recon'); plt.plot(self.loss_history['kl'],label='KL'); plt.legend(); plt.grid(); plt.title('Components')
        betas=[]
        for e in range(1,epochs+1):
            cp=(e%10)/10
            b=min_beta+(max_beta-min_beta)*min(1.0,e/(epochs*0.7))
            betas.append(b*(0.8+0.2*np.sin(cp*2*np.pi)))
        plt.subplot(1,3,3); plt.plot(range(1,epochs+1),betas); plt.grid(); plt.title('Beta')
        plt.tight_layout(); plt.savefig('vae_logs/training_loss.png')
        torch.save({'encoder':self.encoder.state_dict(),
                    'decoder':self.decoder.state_dict(),
                    'epoch':epochs},
                   "vae_logs/audio_vae_final.pt")
