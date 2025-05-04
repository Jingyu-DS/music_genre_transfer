import os
import imageio
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from DataLoading import DataLoad
from EncoderArch import Encoder
from DecoderArch import Decoder
from Loss import get_loss
from TrainerConstruct import Trainer
from datetime import datetime

def prepare_data():
    DATASET_PATH = "./Data/images_original"
    dx, dy = 0, 0
    dimx, dimy = 128, 128
    batch_size = 64
    
    dataload = DataLoad(DATASET_PATH)
    all_photos, all_attrs = dataload.fetch_dataset(dx, dy, dimx, dimy)
    all_photos = np.array(all_photos / 255, dtype='float32')
    
    X_train, X_val = train_test_split(all_photos, test_size=0.1, random_state=365)
    train_loader = DataLoader(dataset=X_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=X_val, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    LATENT_SPACE_SIZE = 128
    
    train_loader, test_loader = prepare_data()
    
    encoder = Encoder(LATENT_SPACE_SIZE)
    decoder = Decoder(LATENT_SPACE_SIZE)
    
    trainer = Trainer(
        trainloader=train_loader,
        testloader=test_loader,
        Encoder=encoder,
        Decoder=decoder,
        latent_dim=LATENT_SPACE_SIZE
    )
    
    trainer.train(num_epochs=50, factor=10)
    timestamp = get_timestamp()
    encoder_name = f"vae_encoder_{timestamp}.pth"
    decoder_name = f"vae_decoder_{timestamp}.pth"
    
    torch.save(encoder.state_dict(), encoder_name)
    torch.save(decoder.state_dict(), decoder_name)
    print(f"Models saved as {encoder_name} and {decoder_name}")

if __name__ == "__main__":
    main()
