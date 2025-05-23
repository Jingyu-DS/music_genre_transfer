import os
import gc
import numpy as np
import torch
import matplotlib.pyplot as plt
import soundfile as sf
from prepare_data import high_quality_mel_to_audio

def memory_efficient_visualize(
    model, data_sample, n_samples=4, sr=22050, save_prefix="memory_efficient_sample"
):
    device = next(model.encoder.parameters()).device
    encoder, decoder = model.encoder, model.decoder

    os.makedirs("audio_results", exist_ok=True)

    indices = np.random.choice(len(data_sample), min(n_samples, len(data_sample)), replace=False)

    fig, axes = plt.subplots(3, n_samples, figsize=(16, 10))
    audio_paths = []
    for i, idx in enumerate(indices):
        x_orig, _ = data_sample[idx]
        x = x_orig.unsqueeze(0).to(device)

        x = 2.0 * ((x - x.min()) / (x.max() - x.min() + 1e-8)) - 1.0

        with torch.no_grad():
            mean, logvar, z = encoder(x)
            x_rec = decoder(z)

            z_random = torch.randn_like(z)
            x_gen = decoder(z_random)

        spec_orig = x_orig.squeeze(0).cpu().numpy()
        spec_rec = x_rec.squeeze().cpu().numpy()
        spec_gen = x_gen.squeeze().cpu().numpy()

        audio_rec = high_quality_mel_to_audio(spec_rec, sr=sr, n_iter=100)
        audio_gen = high_quality_mel_to_audio(spec_gen, sr=sr, n_iter=100)

        rec_path = f"audio_results/{save_prefix}_reconstruction_{i+1}.wav"
        gen_path = f"audio_results/{save_prefix}_generation_{i+1}.wav"
        sf.write(rec_path, audio_rec, sr)
        sf.write(gen_path, audio_gen, sr)
        audio_paths.append(rec_path)

        # plot mel
        axes[0, i].imshow(spec_orig, aspect='auto', origin='lower', cmap='viridis')
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')

        axes[1, i].imshow(spec_rec, aspect='auto', origin='lower', cmap='viridis')
        axes[1, i].set_title(f"Reconstruction {i+1}")
        axes[1, i].axis('off')

        axes[2, i].imshow(spec_gen, aspect='auto', origin='lower', cmap='viridis')
        axes[2, i].set_title(f"Random Generation {i+1}")
        axes[2, i].axis('off')


        del x, mean, logvar, z, x_rec, z_random, x_gen
        torch.cuda.empty_cache()
        gc.collect()

    plt.tight_layout()
    plt.savefig(f'audio_results/{save_prefix}_spectrograms.png', dpi=200)
    plt.close()

    print(f"Visualization complete! Spectrograms and audio have been saved to the audio_results")
    print(f"Path to the first reconstruction audio: {audio_paths[0]}")

    # Play the first reconstructed audio
    try:
        from IPython.display import Audio, display
        print("Playing the first reconstructed audio:")
        audio_file = audio_paths[0]
        display(Audio(audio_file))
    except:
        print("Unable to play audio in this environment. Please open the saved file directly.")

    return audio_paths
