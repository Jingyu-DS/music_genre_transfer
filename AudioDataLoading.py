import os
import librosa
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_audio(path, sr=22050, duration=30):
    try:
        audio, _ = librosa.load(path, sr=sr, duration=duration)
        
        target_length = sr * duration
        
        if len(audio) > target_length:
            audio = audio[:target_length]  
        elif len(audio) < target_length:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        return audio
    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        return None

class DataLoad:
    def __init__(self, data_path):
        self.data_path = data_path
        self.genres = ['blues', 'classical', 'country', 'disco', 
                      'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    def fetch_dataset(self, sr=22050, duration=30):
        audio_paths = []
        genre_labels = []
        
        for genre in self.genres:
            genre_path = os.path.join(self.data_path, genre)
            if not os.path.exists(genre_path):
                continue
                
            for fname in os.listdir(genre_path):
                if fname.endswith(".wav"):
                    fpath = os.path.join(genre_path, fname)
                    audio_paths.append(fpath)
                    genre_labels.append(genre)
        
        df = pd.DataFrame({
            'audio_path': audio_paths,
            'genre': genre_labels
        })
        
        genre_to_num = {genre: i for i, genre in enumerate(self.genres)}
        df['genre_num'] = df['genre'].map(genre_to_num)
        
        all_audio = df['audio_path'].apply(lambda path: load_audio(path, sr=sr, duration=duration))
        
        all_audio = np.stack(all_audio.values)
        all_labels = df['genre_num'].values
        
        return all_audio, all_labels

    def get_genre_names(self):
        return self.genres