import os
import imageio
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_image(path):
    img = imageio.imread(path)
    if img.shape[-1] == 4:  
        img = img[..., :3] 
    return img

class DataLoad:
    def __init__(self, data_path):
        self.data_path = data_path
        self.genres = ['blues', 'classical', 'country', 'disco', 
                      'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    def fetch_dataset(self, dx, dy, dimx, dimy):
        photo_paths = []
        genre_labels = []
        
        for genre in self.genres:
            genre_path = os.path.join(self.data_path, genre)
            if not os.path.exists(genre_path):
                continue
                
            for fname in os.listdir(genre_path):
                if fname.endswith(".png"):
                    fpath = os.path.join(genre_path, fname)
                    photo_paths.append(fpath)
                    genre_labels.append(genre)
        
        df = pd.DataFrame({
            'photo_path': photo_paths,
            'genre': genre_labels
        })
        
        genre_to_num = {genre: i for i, genre in enumerate(self.genres)}
        df['genre_num'] = df['genre'].map(genre_to_num)
        
        all_photos = df['photo_path'].apply(load_image)\
                            .apply(lambda img: img[dy:-dy, dx:-dx] if (dy > 0 and dx > 0) else img)\
                            .apply(lambda img: np.array(Image.fromarray(img).resize([dimx, dimy])))
        
        all_photos = np.stack(all_photos.values).astype('uint8')
        all_labels = df['genre_num'].values
        
        return all_photos, all_labels

    def get_genre_names(self):
        return self.genres


"""
Example Usage:

data_path = "./Data/images_original"
data_loader = DataLoad(data_path)
X, y = data_loader.fetch_dataset(dx=0, dy=0, dimx=128, dimy=128)
genre_names = data_loader.get_genre_names()
"""