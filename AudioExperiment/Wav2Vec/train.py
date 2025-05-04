import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from DataLoading import DataLoad
from Wav2Vec import GenreConditionedW2VEncoder, GenreConditionedDecoder
from TrainerConstruct import GenreConditionedTrainer

def prepare_data(batch_size):
    DATASET_PATH = "./Data/genres_original"

    dataload = DataLoad(DATASET_PATH)
    all_audios, all_attrs = dataload.fetch_dataset()

    print(all_audios.shape)
    print(all_attrs.shape)
    print("NaNs in audios:", np.isnan(all_audios).any())
    print("Infs in audios:", np.isinf(all_audios).any())
    print("Max audio value:", all_audios.max())
    print("Min audio value:", all_audios.min())

    all_audios = torch.from_numpy(all_audios).float().unsqueeze(1)  # (B, 1, T)
    all_attrs = torch.from_numpy(all_attrs).long()                  # (B,)

    X_train, X_val, y_train, y_val = train_test_split(
        all_audios, all_attrs, test_size=0.1, random_state=365, stratify=all_attrs
    )

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    LATENT_SPACE_SIZE = 256
    
    train_loader, test_loader = prepare_data(32)
    
    encoder = GenreConditionedW2VEncoder(LATENT_SPACE_SIZE)
    decoder = GenreConditionedDecoder(LATENT_SPACE_SIZE)
    
    trainer = GenreConditionedTrainer(
        trainloader=train_loader,
        testloader=test_loader,
        Encoder=encoder,
        Decoder=decoder,
        latent_dim=LATENT_SPACE_SIZE,
        lr= 1e-4
    )
    
    trainer.train(num_epochs=50, factor=10)
    timestamp = get_timestamp()
    encoder_name = f"audio_Wav2Vec_encoder_{timestamp}.pth"
    decoder_name = f"audio_Wav2Vec_decoder_{timestamp}.pth"
    
    torch.save(encoder.state_dict(), encoder_name)
    torch.save(decoder.state_dict(), decoder_name)
    print(f"Models saved as {encoder_name} and {decoder_name}")

if __name__ == "__main__":
    main()