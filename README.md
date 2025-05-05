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
prepare_data.py implements load_audio(), enhanced_audio_to_mel() and prepare_enhanced_dataset() to turn your WAV files into 1×128×512 Mel-spectrogram tensors and build train/validation DataLoaders. model.py defines LightResBlock, MemoryEfficientEncoder and MemoryEfficientDecoder. trainer.py brings these together in EfficientLoss and MemoryEfficientTrainer, handling the 0.7 MSE+0.3 grad+β·KL loss, cosine-annealing, gradient clipping, accumulation and checkpointing to vae_logs/. audio_io.py contains high_quality_mel_to_audio(), enhanced_visualize_and_play() and interpolate_latent_space() to convert spectrograms back to WAVs and save figures/audios under audio_results/.
