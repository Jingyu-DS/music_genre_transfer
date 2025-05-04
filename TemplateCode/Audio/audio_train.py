import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from AudioDataLoading import DataLoad
from AudioEncoderArch import AudioEncoder
from AudioDecoderArch import AudioDecoder
from Loss import get_loss
from AudioTrainerConstruct import Trainer
from datetime import datetime

def prepare_data():
    DATASET_PATH = "./Data/genres_original"
    batch_size = 32
    
    dataload = DataLoad(DATASET_PATH)
    all_audios, all_attrs = dataload.fetch_dataset()
    all_audios = torch.from_numpy(all_audios).float()
    all_attrs = torch.from_numpy(all_attrs).long()
    
    all_audios = all_audios.unsqueeze(1)  
    
    X_train, X_val = train_test_split(all_audios, test_size=0.1, random_state=365)
    train_loader = DataLoader(dataset=X_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=X_val, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    LATENT_SPACE_SIZE = 128
    
    train_loader, test_loader = prepare_data()
    
    encoder = AudioEncoder(LATENT_SPACE_SIZE)
    decoder = AudioDecoder(LATENT_SPACE_SIZE)
    
    trainer = Trainer(
        trainloader=train_loader,
        testloader=test_loader,
        Encoder=encoder,
        Decoder=decoder,
        latent_dim=LATENT_SPACE_SIZE
    )
    
    trainer.train(num_epochs=50, factor=10)
    timestamp = get_timestamp()
    encoder_name = f"audio_vae_encoder_{timestamp}.pth"
    decoder_name = f"audio_vae_decoder_{timestamp}.pth"
    
    torch.save(encoder.state_dict(), encoder_name)
    torch.save(decoder.state_dict(), decoder_name)
    print(f"Models saved as {encoder_name} and {decoder_name}")

if __name__ == "__main__":
    main()
