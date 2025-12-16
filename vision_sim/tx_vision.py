
import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image

# Add parent directory to path to import models
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from models.model import SemViT_Encoder_Only
from config.usrp_config import NORMALIZE_CONSTANT

# Configuration matching server.py
ARCH = ['C', 'C', 'V', 'V', 'C', 'C']
FILTERS = [256, 256, 256, 256, 256, 256]
REPETITIONS = [1, 1, 3, 3, 1, 1]
NUM_SYMBOLS = 512
HAS_GDN = False

def main():
    print("Initializing Encoder...")
    encoder = SemViT_Encoder_Only(
        ARCH,
        FILTERS,
        REPETITIONS,
        has_gdn=HAS_GDN,
        num_symbols=NUM_SYMBOLS
    )
    
    # Load CIFAR-10 data
    print("Loading CIFAR-10 image...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Select a random image from test set
    idx = np.random.randint(0, len(x_test))
    image = x_test[idx]
    
    # Save original image for comparison
    Image.fromarray(image).save("vision_sim/original_img.png")
    print(f"Saved original image to vision_sim/original_img.png (Class: {y_test[idx]})")
    
    # Preprocess
    image = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0
    image = tf.reshape(image, (1, 32, 32, 3))
    
    # Run Encoder - this builds the model with the input shape
    print("Encoding...")
    # We need to call it once to build weights if not built, but Keras Functional/Subclass should handle it?
    # actually SemViT_Encoder_Only.call calls self.encoder(x)
    
    # Extract IQ symbols
    # Output of encoder is (batch, num_symbols, 2)
    # But wait, looking at model.py:
    # SemViT_Encoder.call returns tf.reshape(x, (-1, h*w*c//2, 2))
    # Let's check the shape.
    
    iq_symbols = encoder(image)
    print(f"Encoder Output Shape: {iq_symbols.shape}")
    
    # Flatten to interleaved IQ
    # Server.py does: 
    # i = data[:,:,0].numpy().flatten()
    # q = data[:,:,1].numpy().flatten()
    # But we want interleaved for generic file source usually, or we can just send I then Q?
    # The plan said "Interleaved Complex Float (Real, Imag, Real, Imag...)".
    # Let's see how `usrp/server.py` sent it. It sent `constellations.npz`.
    # But `utils/usrp_utils.py` `to_constellation_array` seems to pack it for LabVIEW.
    # Here we are simulating a "File Source" in GNU Radio which usually expects Complex64 (Interleaved I, Q).
    
    i = iq_symbols[:,:,0].numpy().flatten()
    q = iq_symbols[:,:,1].numpy().flatten()
    
    # Normalization (Power constraint simulation)
    # In server.py: i = np.round(np.clip(i / NORMALIZE_CONSTANT * 32767, ...))
    # This was for USRP DAC.
    # For simulation, we can just keep them as floats.
    # However, let's normalize to unit power or similar for the channel.
    # Or just leave as is since we control the noise level.
    # Let's save as float32 interleaved.
    
    interleaved_iq = np.zeros(len(i) * 2, dtype=np.float32)
    interleaved_iq[0::2] = i
    interleaved_iq[1::2] = q
    
    # Save to bin
    with open("vision_sim/iq_tx.bin", "wb") as f:
        interleaved_iq.tofile(f)
        
    print(f"Saved {len(interleaved_iq)//2} complex symbols to vision_sim/iq_tx.bin")

if __name__ == "__main__":
    main()
