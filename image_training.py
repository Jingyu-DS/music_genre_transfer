from image_prepare_data import prepare_dataset
from image_encoder_dncoder import MemoryEfficientEncoder, MemoryEfficientDecoder
from image_final_trainer import MemoryEfficientTrainer
from image_final_visualization import memory_efficient_visualize
import os
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split



def main():
    root = "/content/Data/genres_original"
    genres = ['blues','classical','country','disco','hiphop',
              'jazz','metal','pop','reggae','rock']
    batch_size = 4
    latent_dim = 512
    epochs = 50
    sr = 22050


    X, y = prepare_dataset(root, genres, duration=10, sr=sr)

    # train/test
    idx_tr, idx_va = train_test_split(
        np.arange(len(X)), test_size=0.1, stratify=y, random_state=42
    )
    train_ds = TensorDataset(X[idx_tr], y[idx_tr])
    val_ds = TensorDataset(X[idx_va], y[idx_va])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           pin_memory=True, num_workers=0)

    print(f"Train batches: {len(train_loader)}, "
          f"One batch X shape: {next(iter(train_loader))[0].shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    encoder = MemoryEfficientEncoder(in_channels=1, latent_dim=latent_dim)
    decoder = MemoryEfficientDecoder(latent_dim=latent_dim, out_channels=1)

    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters: {encoder_params + decoder_params:,}")


    torch.cuda.empty_cache()
    gc.collect()


    trainer = MemoryEfficientTrainer(train_loader, val_loader, encoder, decoder, device=device)


    trainer.train(epochs=epochs,
                  warmup=5,
                  max_beta=0.01,
                  min_beta=0.001,
                  accumulation_steps=8)


    audio_paths = memory_efficient_visualize(trainer, val_ds, n_samples=4)

    print("finish! save to 'audio_results)

    return trainer

if __name__ == "__main__":
    trainer = main()
