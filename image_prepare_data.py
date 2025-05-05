import numpy as np
import librosa
import soundfile as sf
import os
import torch


def load_audio(path, sr=22050, duration=10, normalize=True):
    wav, _ = librosa.load(path, sr=sr, duration=duration)
    tgt_len = sr * duration
    if len(wav) < tgt_len:
        wav = np.pad(wav, (0, tgt_len-len(wav)), mode='reflect')
    else:
        wav = wav[:tgt_len]
    if normalize:
        wav = wav / (np.max(np.abs(wav)) + 1e-8)

    return wav

def enhanced_audio_to_mel(
    wav, sr=22050, n_fft=2048, hop_length=512, n_mels=128,
    target_frames=512, db_scale=True, top_db=80.0
):
    """generate mel"""
    M = librosa.feature.melspectrogram(
        y=wav, sr=sr,
        n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, power=2.0,
        fmin=20, fmax=8000
    )

    if db_scale:
        M = librosa.power_to_db(M, ref=np.max, top_db=top_db)


        M_min, M_max = M.min(), M.max()
        M = 2.0 * ((M - M_min) / (M_max - M_min + 1e-8)) - 1.0


    M = librosa.util.fix_length(M, size=target_frames, axis=1)

    return M

def high_quality_mel_to_audio(
    mel_spec, sr=22050, n_fft=2048, hop_length=512,
    n_iter=100, power=2.0, ref_db=20.0, max_db=80.0
):
    if mel_spec.min() < 0:
        mel_spec = (mel_spec + 1) / 2
        mel_spec = mel_spec * max_db - ref_db


    mel_power = librosa.db_to_power(mel_spec)


    wav = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        power=power,
        n_iter=n_iter,
        fmin=20,
        fmax=8000
    )

    wav = np.sign(wav) * np.log(1 + 9 * np.abs(wav)) / np.log(10)
    wav = wav / (np.max(np.abs(wav)) + 1e-8) * 0.9

    return wav

def prepare_dataset(root, genres, duration=10, sr=22050):
    audios, labels = [], []
    for gi, g in enumerate(genres):
        folder = os.path.join(root, g)
        if not os.path.isdir(folder): continue
        for fn in os.listdir(folder):
            if fn.lower().endswith(".wav"):
                try:
                    wav = load_audio(os.path.join(folder, fn), sr=sr, duration=duration)
                    M = enhanced_audio_to_mel(wav, sr=sr, target_frames=512)
                    audios.append(M)
                    labels.append(gi)
                except Exception as e:
                    print(f"Error processing {fn}: {e}")
                    continue

    audios = np.stack(audios)
    X = torch.tensor(audios, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)

    print("Dataset shapes:", X.shape, y.shape)
    return X, y
