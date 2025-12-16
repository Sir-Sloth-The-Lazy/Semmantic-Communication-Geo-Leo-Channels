
import os
import subprocess
import sys

def run_command(command):
    print(f"Running: {command}")
    ret = subprocess.call(command, shell=True)
    if ret != 0:
        print(f"Command failed with exit code {ret}")
        sys.exit(ret)

def main():
    print("=== Starting Vision Transformer Semantic Communication Simulation ===")
    
    # 1. Transmitter
    print("\n[Step 1] Transmitter")
    run_command(f"/opt/homebrew/Caskroom/miniconda/base/envs/deepsc_env/bin/python vision_sim/tx_vision.py")
    
    # 2. Channel
    print("\n[Step 2] Wireless Channel Simulation")
    run_command(f"/opt/homebrew/Caskroom/miniconda/base/envs/deepsc_env/bin/python vision_sim/channel_sim.py --snr 10")
    
    # 3. Receiver
    print("\n[Step 3] Receiver")
    run_command(f"/opt/homebrew/Caskroom/miniconda/base/envs/deepsc_env/bin/python vision_sim/rx_vision.py")
    
    print("\n=== Simulation Complete ===")

if __name__ == "__main__":
    main()
