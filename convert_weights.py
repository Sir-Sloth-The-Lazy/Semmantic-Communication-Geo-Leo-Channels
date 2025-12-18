import os
import tensorflow as tf
from models.model import SemViT

def main():
    # Configuration matches the AWGN training parameters
    # train_dist.py 512 AWGN 10 CCVVCC ...
    DATA_SIZE = 512
    SNR_DB = 10
    BLOCK_TYPES = ['C', 'C', 'V', 'V', 'C', 'C']
    FILTERS = [256, 256, 256, 256, 256, 256]
    REPETITIONS = [1, 1, 3, 3, 1, 1]
    
    # Paths
    # The checkpoint path (prefix for .data and .index)
    checkpoint_path = 'weights/CCVVCC_512_10dB_599'
    # Target output path
    output_path = 'weights/CCVVCC_512_10dB_599.weights.h5'
    
    print("Building model...")
    # Build model (Channel AWGN doesn't affect weight loading, but good for consistency)
    model = SemViT(
        block_types=BLOCK_TYPES,
        filters=FILTERS,
        num_blocks=REPETITIONS,
        has_gdn=True,
        num_symbols=DATA_SIZE,
        snrdB=SNR_DB,
        channel='AWGN'
    )
    
    # Compile to ensure build (metrics can be dummy)
    model.compile(optimizer='adam', loss='mse')
    
    # Build input shape (CIFAR-10)
    model(tf.zeros((1, 32, 32, 3)))
    
    print(f"Loading weights from {checkpoint_path}...")
    try:
        model.load_weights(checkpoint_path)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    print(f"Saving weights to {output_path}...")
    model.save_weights(output_path)
    print("Done.")

if __name__ == "__main__":
    main()
