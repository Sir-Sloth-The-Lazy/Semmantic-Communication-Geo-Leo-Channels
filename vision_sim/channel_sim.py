
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snr', type=float, default=10.0, help='SNR in dB')
    args = parser.parse_args()
    
    input_file = "vision_sim/iq_tx.bin"
    output_file = "vision_sim/iq_rx.bin"
    
    print(f"Channel Simulation: SNR = {args.snr} dB")
    
    # Load IQ
    try:
        data = np.fromfile(input_file, dtype=np.float32)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Run tx_vision.py first.")
        return

    # Convert to complex
    complex_data = data[0::2] + 1j * data[1::2]
    
    # Calculate Signal Power
    current_power = np.mean(np.abs(complex_data)**2)
    print(f"Signal Power: {current_power:.4f}")
    
    # Calculate Noise Power needed for target SNR
    # SNR_dB = 10 * log10(P_signal / P_noise)
    # P_noise = P_signal / (10^(SNR_dB/10))
    noise_power = current_power / (10**(args.snr/10))
    noise_std = np.sqrt(noise_power / 2) # /2 because complex noise splits power into I and Q
    
    print(f"Adding AWGN with Noise Power: {noise_power:.4f}")

    # Generate Noise
    noise = noise_std * (np.random.randn(len(complex_data)) + 1j * np.random.randn(len(complex_data)))
    
    # Add Noise
    noisy_data = complex_data + noise
    
    # Save as Interleaved float32
    output_data = np.zeros(len(data), dtype=np.float32)
    output_data[0::2] = np.real(noisy_data)
    output_data[1::2] = np.imag(noisy_data)
    
    with open(output_file, "wb") as f:
        output_data.tofile(f)
        
    print(f"Saved noisy IQ to {output_file}")

if __name__ == "__main__":
    main()
