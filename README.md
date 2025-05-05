## Music Genre Transfer
#### Utilized Dataset
We use the GTZAN database, a widely used benchmark for music genre classification. The database consists of 1,000 30-second audio pieces with a balanced choice of 10 musical genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, and rock. The original recordings were gathered from a wide collection of sources, including CD tracks, radio, and microphone recordings, thereby reflecting a wide range of recording conditions. The data can be downloaded here: 
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data

#### Components of the Repo
TemplateCode - This folder contains the base model used for general pipeline set up, including fundemental data loading, base model architecture, trainer structure and training pipeline set up. 

AudioExperiment - This folder contains all different model architecture we have explored, mainly from popular relevant paper reading, including conditional VAE with music genre encoded, audio model with attention machenism, audio model with residual block, audio wavenet, and transfermer based model like wav2vec. All of them are experiemented, comparing to each other using the loss and display on the latent space. 

ImageExperiment - This folder contains all different model architecture we have explored on audio converted images, including hierarchical encoder and CNN-based model with different layers added with hyperparameters fine tuning. 

LatentSpaceInterpretation - This folder contains the work we have done to reduce the dimensionality of representations learnt to see how separate audios from different genres are to provide us with better understanding of the distributions of latent bottleneck of VAEs. It helps us monitor whether VAEs get lazy during the training or not. 

Outside the folders are the final version of models that we will introduce how to run it below. 


### Running the audio model and conduct genre transfer
**SlicingDataLoading.py** contains the data loader module. **audio_resnet_encoder_wavenet_decoder.py** contains the architecture of final version of model. 

If you need to re-run the training process, just open the **audio_model_training.ipynb**, run the cells there. The final model will be saved in the same repo as files ending as pth. 

Then, it is ready to do the genre transfer of the music. You can open **audio_genre_transfer_inference.ipynb** and change the paths of model to the pth files you saved. Then, run the cells, the bottom one will give you the transferred mel spectragram of five samples and one sample of audio after the genre transfer. 


### Running the Spectrogram VAE End-to-End
The codebase is structured into several modular files:

- **image_encoder_decoder.py**: Contains the VAE architecture with `MemoryEfficientEncoder` and `MemoryEfficientDecoder` classes
- **image_final_loss.py**: Implements the `EfficientLoss` class for training the VAE
- **image_final_trainer.py**: Contains the `MemoryEfficientTrainer` class that manages the training process
- **image_final_visualization.py**: Functions for visualizing model outputs and generating audio samples
- **image_prepare_data.py**: Utilities for audio processing, including mel-spectrogram conversion
- **image_training(runner).py**: Main script that orchestrates the training workflow

## Running the Audio VAE Model

### Training the Model
1. Ensure you have the required dependencies installed (PyTorch, librosa, matplotlib, numpy, soundfile)
2. To train the model from scratch, simply run: “python image_training.py”
3. This will:
- Load and process audio data from the specified genre folders
- Initialize the encoder and decoder models
- Train the VAE with memory-efficient techniques
- Save checkpoints to the `vae_logs` directory

### Visualization and Audio Generation

The training process automatically generates:
- Loss plots saved to `vae_logs/training_loss.png`
- Audio reconstructions and random generations saved to the `audio_results` directory
- Mel-spectrogram visualizations comparing original, reconstructed, and randomly generated samples

### Model Customization

You can modify key parameters in the `image_training.py` file:
- `root`: Path to your audio dataset
- `genres`: List of genre folders to include
- `batch_size`: Training batch size (smaller values use less memory)
- `latent_dim`: Size of the latent space representation
- `epochs`: Number of training epochs

The model is designed with memory efficiency in mind, using gradient accumulation and periodic garbage collection to handle larger audio datasets even with limited GPU resources.
