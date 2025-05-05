import numpy as np
import librosa
import soundfile as sf


def audio_to_mel(
    wav_path: str,
    sr: int = 22050,
    duration: float = 10.0,
    normalize: bool = True,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    target_frames: int = 512,
    db_scale: bool = True,
    top_db: float = 80.0
) -> np.ndarray:
    """
    Load an audio file and convert it to a normalized Mel-spectrogram.

    Args:
        wav_path: Path to the input WAV file.
        sr: Sampling rate.
        duration: Duration (in seconds) to load.
        normalize: Whether to normalize waveform to [-1, 1].
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        n_mels: Number of Mel bands.
        target_frames: Number of time frames to pad/trim to.
        db_scale: Whether to convert power to dB scale.
        top_db: Threshold for dB conversion.

    Returns:
        A Mel-spectrogram array of shape (n_mels, target_frames) in [-1, 1].
    """
    wav, _ = librosa.load(wav_path, sr=sr, duration=duration)
    tgt_len = int(sr * duration)
    if len(wav) < tgt_len:
        wav = np.pad(wav, (0, tgt_len - len(wav)), mode='reflect')
    else:
        wav = wav[:tgt_len]
    if normalize:
        wav = wav / (np.max(np.abs(wav)) + 1e-8)

    M = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
        fmin=20,
        fmax=8000
    )
    if db_scale:
        M = librosa.power_to_db(M, ref=np.max, top_db=top_db)
        M_min, M_max = M.min(), M.max()
        M = 2.0 * ((M - M_min) / (M_max - M_min + 1e-8)) - 1.0
    M = librosa.util.fix_length(M, size=target_frames, axis=1)
    return M


def mel_to_audio(
    mel_spec: np.ndarray,
    output_path: str,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_iter: int = 150,
    power: float = 2.0,
    ref_db: float = 20.0,
    max_db: float = 80.0
) -> np.ndarray:
    """
    Convert a normalized Mel-spectrogram back to waveform and save to file.

    Args:
        mel_spec: Input Mel-spectrogram in [-1, 1].
        output_path: Path to save the reconstructed WAV.
        sr: Sampling rate.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        n_iter: Number of Griffin-Lim iterations.
        power: Power exponent for inversion.
        ref_db: Reference dB level.
        max_db: Maximum dB level.

    Returns:
        The reconstructed waveform as a numpy array.
    """
    spec = mel_spec.copy()
    if spec.min() < 0:
        spec = (spec + 1) / 2
        spec = spec * max_db - ref_db
    mel_power = librosa.db_to_power(spec)
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
    sf.write(output_path, wav, sr)
    return wav
