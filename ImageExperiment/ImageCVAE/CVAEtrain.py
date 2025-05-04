def main():
    LATENT_SPACE_SIZE = 128
    CONDITION_DIM = 10  # For example, size of your one-hot genre vector
    
    # Prepare your training and test DataLoaders, which should now yield (x, condition) tuples.
    train_loader, test_loader = prepare_data()  # Make sure your DataLoad function is updated
    
    # Instantiate the CVAE components (note: these classes now expect the additional condition info).
    from EncoderArchCVAE import CVAE_Encoder  # Assuming you've defined a CVAE encoder as shown earlier.
    from DecoderArchCVAE import CVAE_Decoder  # Similar for the decoder.
    
    encoder = CVAE_Encoder(latent_dim=LATENT_SPACE_SIZE, condition_dim=CONDITION_DIM, input_shape=(3, 128, 128), use_embedding=False)
    decoder = CVAE_Decoder(latent_dim=LATENT_SPACE_SIZE, condition_dim=CONDITION_DIM)
    
    trainer = TrainerCVAE(
        trainloader=train_loader,
        testloader=test_loader,
        Encoder=encoder,
        Decoder=decoder,
        latent_dim=LATENT_SPACE_SIZE,
        device="cuda"
    )
    
    trainer.train(num_epochs=50, factor=100)
    
    # Optionally, save the trained models with a timestamp.
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(encoder.state_dict(), f"cvae_encoder_{timestamp}.pth")
    torch.save(decoder.state_dict(), f"cvae_decoder_{timestamp}.pth")
    print(f"Models saved as cvae_encoder_{timestamp}.pth and cvae_decoder_{timestamp}.pth")

if __name__ == "__main__":
    main()
