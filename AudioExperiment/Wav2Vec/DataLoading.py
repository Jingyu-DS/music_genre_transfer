import os
import librosa
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoad:
    def __init__(self, data_path):
        self.data_path = data_path
        self.genres = ['blues', 'classical', 'country', 'disco',
                       'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    def fetch_dataset(self, sr=16000, segment_duration=5, apply_augmentation=False):
        audio_segments = []
        genre_labels = []

        for genre in self.genres:
            genre_path = os.path.join(self.data_path, genre)
            if not os.path.exists(genre_path):
                continue

            for fname in os.listdir(genre_path):
                if fname.endswith(".wav"):
                    fpath = os.path.join(genre_path, fname)

                    # Load full 30s audio
                    full_audio, _ = librosa.load(fpath, sr=sr, duration=30)
                    total_length = sr * 30  # 30 seconds audio length
                    segment_length = sr * segment_duration  # segment length in samples
                    num_segments = total_length // segment_length

                    # Explicitly create segments
                    for i in range(num_segments):
                        start_sample = int(i * segment_length)
                        end_sample = int(start_sample + segment_length)

                        segment_audio = full_audio[start_sample:end_sample]

                        # Check segment length explicitly
                        if len(segment_audio) < segment_length:
                            padding = segment_length - len(segment_audio)
                            segment_audio = np.pad(segment_audio, (0, padding), mode='constant')

                        # Normalization explicitly
                        max_val = np.abs(segment_audio).max()
                        if max_val > 0:
                            segment_audio = segment_audio / max_val

                        # Augmentation explicitly (optional for training set)
                        if apply_augmentation:
                            spectrogram = librosa.stft(segment_audio, n_fft=1024, hop_length=256)
                            spectrogram_mag = np.abs(spectrogram)

                            spec_tensor = torch.tensor(spectrogram_mag).unsqueeze(0)
                            freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
                            spec_tensor = freq_mask(spec_tensor)
                            time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)
                            spec_tensor = time_mask(spec_tensor)

                            augmented_spec = spec_tensor.squeeze(0).numpy()
                            segment_audio = librosa.istft(
                                augmented_spec * np.exp(1j * np.angle(spectrogram)),
                                hop_length=256
                            )

                            # Renormalize after augmentation
                            max_aug_val = np.abs(segment_audio).max()
                            if max_aug_val > 0:
                                segment_audio = segment_audio / max_aug_val

                            # Ensure exact length after ISTFT
                            if len(segment_audio) > segment_length:
                                segment_audio = segment_audio[:segment_length]
                            elif len(segment_audio) < segment_length:
                                segment_audio = np.pad(segment_audio, (0, segment_length - len(segment_audio)), mode='constant')

                        audio_segments.append(segment_audio)
                        genre_labels.append(genre)

        audio_segments = np.stack(audio_segments)
        genre_to_num = {genre: i for i, genre in enumerate(self.genres)}
        all_labels = np.array([genre_to_num[g] for g in genre_labels])

        return audio_segments, all_labels