import numpy as np
import torch
import librosa
import soundfile as sf
from IPython.display import Audio


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


def enhanced_audio_to_mel(wav, sr=22050, n_fft=2048, hop_length=512,
                          n_mels=128, target_frames=512, db_scale=True, top_db=80.0):
    M = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, power=2.0, fmin=20, fmax=8000
    )
    if db_scale:
        M = librosa.power_to_db(M, ref=np.max, top_db=top_db)
        M = 2.0*((M - M.min())/(M.max() - M.min()+1e-8)) - 1.0
    return librosa.util.fix_length(M, size=target_frames, axis=1)


def high_quality_mel_to_audio(mel_spec, sr=22050, n_fft=2048, hop_length=512,
                              n_iter=150, power=2.0, ref_db=20.0, max_db=80.0):
    if mel_spec.min() < 0:
        mel_spec = (mel_spec+1)/2*max_db - ref_db
    mel_power = librosa.db_to_power(mel_spec)
    wav = librosa.feature.inverse.mel_to_audio(
        mel_power, sr=sr, n_fft=n_fft, hop_length=hop_length,
        power=power, n_iter=n_iter, fmin=20, fmax=8000
    )
    wav = np.sign(wav)*np.log(1+9*np.abs(wav))/np.log(10)
    return wav/np.max(np.abs(wav))*0.9
