
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.model import SemViT_Decoder_Only
from models.metrics import psnr, ssim

# Configuration matching server.py
ARCH = ['C', 'C', 'V', 'V', 'C', 'C']
FILTERS = [256, 256, 256, 256, 256, 256]
REPETITIONS = [1, 1, 3, 3, 1, 1]
NUM_SYMBOLS = 512
HAS_GDN = False

def main():
    print("Initializing Decoder...")
    decoder = SemViT_Decoder_Only(
        ARCH,
        FILTERS,
        REPETITIONS,
        has_gdn=HAS_GDN,
        num_symbols=NUM_SYMBOLS
    )
    
    # Load RX IQ
    input_file = "vision_sim/iq_rx.bin"
    try:
        data = np.fromfile(input_file, dtype=np.float32)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Run channel_sim.py first.")
        return
        
    # Reshape to (Batch, Num_Symbols, 2)
    # We transmitted 512 symbols
    # data length should be 1024 floats
    expected_len = NUM_SYMBOLS * 2
    if len(data) != expected_len:
        print(f"Warning: Data length {len(data)} does not match expected {expected_len}. Truncating or padding.")
        if len(data) > expected_len:
            data = data[:expected_len]
        else:
            data = np.pad(data, (0, expected_len - len(data)))
            
    iq_reshaped = data.reshape(1, NUM_SYMBOLS, 2)
    
    print("Decoding...")
    # Decode
    # Input to decoder is (batch, num_symbols, 2)
    reconstructed_img = decoder(iq_reshaped)
    
    # Post-process
    # Output is (1, 32, 32, 3) in [0, 1] usually (Sigmoid activation in model.py)
    # Let's verify activation in model.py -> 'sigmoid'. Correct.
    
    reconstructed_np = reconstructed_img[0].numpy()
    reconstructed_uint8 = (reconstructed_np * 255).astype(np.uint8)
    
    Image.fromarray(reconstructed_uint8).save("vision_sim/decoded_img.png")
    print("Saved decoded image to vision_sim/decoded_img.png")
    
    # Metrics
    if os.path.exists("vision_sim/original_img.png"):
        original_pil = Image.open("vision_sim/original_img.png")
        original_np = np.array(original_pil).astype(np.float32) / 255.0
        
        # PSNR
        # PSNR & SSIM using standardized metrics
        # Note: psnr/ssim functions in metrics.py expect [0,1]
        psnr_val = psnr(original_np, reconstructed_np)
        ssim_val = ssim(original_np, reconstructed_np)
        
        print("-" * 30)
        print(f"Evaluation Results:")
        print(f"PSNR: {psnr_val.numpy():.4f} dB")
        print(f"SSIM: {ssim_val.numpy():.4f}")
        print("-" * 30)
    else:
        print("Original image not found, skipping metrics.")

if __name__ == "__main__":
    main()
