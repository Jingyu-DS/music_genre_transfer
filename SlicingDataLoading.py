import os
import librosa
import pandas as pd
import numpy as np

def load_audio(path, sr=18000, duration=5):
    try:
        original_duration = 30
        offset_start = 2.5  # skip first 2.5 sec
        offset_end = original_duration - 2.5  # skip last 2.5 sec

        segment_duration = duration  # 5 seconds per segment
        segments = []

        # Starting points of each segment (5 segments total)
        offsets = np.linspace(offset_start, offset_end - segment_duration, num=5)

        for offset in offsets:
            audio, _ = librosa.load(path, sr=sr, duration=segment_duration, offset=offset)

            target_length = int(sr * duration)
            if len(audio) > target_length:
                audio = audio[:target_length]
            elif len(audio) < target_length:
                padding = target_length - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')

            # Normalize
            # max_val = np.abs(audio).max()
            # audio = audio / max_val if max_val > 0 else audio

            segments.append(audio)

        return segments

    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        return []

class DataLoad:
    def __init__(self, data_path):
        self.data_path = data_path
        self.genres = ['blues', 'classical', 'country', 'disco',
                       'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    def fetch_dataset(self, sr=18000, duration=5):
        audio_segments = []
        genre_labels = []

        for genre in self.genres:
            genre_path = os.path.join(self.data_path, genre)
            if not os.path.exists(genre_path):
                continue

            for fname in os.listdir(genre_path):
                if fname.endswith(".wav"):
                    fpath = os.path.join(genre_path, fname)
                    segments = load_audio(fpath, sr=sr, duration=duration)
                    audio_segments.extend(segments)
                    genre_labels.extend([genre] * len(segments))

        df = pd.DataFrame({
            'audio': audio_segments,
            'genre': genre_labels
        })

        genre_to_num = {genre: i for i, genre in enumerate(self.genres)}
        df['genre_num'] = df['genre'].map(genre_to_num)

        all_audio = np.stack(df['audio'].values)
        all_labels = df['genre_num'].values

        return all_audio, all_labels

    def get_genre_names(self):
        return self.genres