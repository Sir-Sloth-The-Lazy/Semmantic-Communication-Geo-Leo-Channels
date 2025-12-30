#!/usr/bin/env python3

"""
USRP B200 Driver for Semantic Communications Project
This script bridges the TCP connection from the client to the physical USRP hardware.
"""

import sys
import os
import socket
import time
import signal
import numpy as np
import threading

# Append parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

try:
    import uhd
except ImportError:
    print("Error: The 'uhd' library is not installed. Please install it to use the USRP B200.")
    print("You may need to install the UHD driver and python bindings.")
    sys.exit(1)

from config.usrp_config import CLIENT_ADDR, CLIENT_PORT
from utils.networking import receive_constellation_tcp, METADATA_BYTES
from usrp.pilot import SAMPLE_SIZE

# USRP Configuration Defaults for B200
USRP_ARGS = "type=b200"
CENTER_FREQ = 2.4e9  # 2.4 GHz
SAMPLE_RATE = 1e6    # 1 Msps
TX_GAIN = 50         # dB
RX_GAIN = 50         # dB
BANDWIDTH = 1e6      # 1 MHz

def setup_usrp(args=USRP_ARGS, rate=SAMPLE_RATE, center_freq=CENTER_FREQ, tx_gain=TX_GAIN, rx_gain=RX_GAIN, bw=BANDWIDTH):
    """Configures the USRP device for Tx and Rx."""
    print(f"initializing USRP with args: {args}")
    usrp = uhd.usrp.MultiUSRP(args)

    # Validate device
    if usrp.get_num_mboards() == 0:
        print("No USRP devices found!")
        sys.exit(1)

    # Set clock and time
    usrp.set_clock_source("internal")
    # usrp.set_time_source("internal") # Optional, depending on need

    # Configure Sample Rate
    print(f"Setting sample rate to {rate/1e6} Msps...")
    usrp.set_tx_rate(rate)
    usrp.set_rx_rate(rate)

    # Configure Carrier Frequency
    print(f"Setting center frequency to {center_freq/1e9} GHz...")
    usrp.set_tx_freq(uhd.types.TuneRequest(center_freq))
    usrp.set_rx_freq(uhd.types.TuneRequest(center_freq))

    # Configure Gain
    print(f"Setting Tx Gain: {tx_gain} dB, Rx Gain: {rx_gain} dB...")
    usrp.set_tx_gain(tx_gain)
    usrp.set_rx_gain(rx_gain)

    # Configure Bandwidth
    print(f"Setting Bandwidth to {bw/1e6} MHz...")
    usrp.set_tx_bandwidth(bw)
    usrp.set_rx_bandwidth(bw)

    # Verify settings
    actual_tx_rate = usrp.get_tx_rate()
    actual_rx_rate = usrp.get_rx_rate()
    print(f"Actual Tx Rate: {actual_tx_rate}")
    print(f"Actual Rx Rate: {actual_rx_rate}")

    return usrp

def transmit_worker(usrp, tx_streamer, data_queue, stop_event):
    """Thread to handle transmission."""
    metadata = uhd.types.TXMetadata()
    
    while not stop_event.is_set():
        if not data_queue:
            time.sleep(0.001)
            continue
            
        # Get data from queue (expecting complex64 numpy array)
        if len(data_queue) > 0:
            tx_data = data_queue.pop(0) # Simple list used as queue for simplicity
            
            # Send to USRP
            # data shape should be (1, N) for single channel
            if len(tx_data.shape) == 1:
                tx_data = tx_data.reshape(1, -1)
            
            num_samps = tx_data.shape[1]
            
            # Metadata timestamping can be added here if needed
            metadata.start_of_burst = True
            metadata.end_of_burst = True
            
            samples_sent = tx_streamer.send(tx_data, metadata)
            if samples_sent != num_samps:
                print(f"Warning: Sent {samples_sent}/{num_samps} samples")

def receive_and_forward(usrp, rx_streamer, client_socket, stop_event, num_samples):
    """Receives data from USRP and forwards to TCP client."""
    metadata = uhd.types.RXMetadata()
    
    # Pre-allocate buffer
    recv_buffer = np.zeros((1, num_samples), dtype=np.complex64)
    
    # Issue stream command
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = num_samples
    stream_cmd.stream_now = True
    
    rx_streamer.issue_stream_cmd(stream_cmd)
    
    # Receive
    samples_received = rx_streamer.recv(recv_buffer, metadata)
    
    if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
        print(f"Error receiving samples: {metadata.strerror()}")
        return None
        
    # Process received data for TCP (convert back to interleaved I/Q int32 format expected by Utils?)
    # Wait, client expects 'send_data' which is `len + iq_concat`
    # server side: 
    # i = np.right_shift(np.left_shift(data, 16), 16).astype('>f4') / 32767
    
    # We need to reverse the conversion done in `server.py` and `mockup_usrp.py`?
    # mockup_usrp.py sends: 
    # iq_concat = np.concatenate((i, q), axis=-1).astype('>f4')
    # send_data = len(iq_concat).to_bytes(length=METADATA_BYTES, byteorder='big') + iq_concat.tobytes()
    
    # In `mockup_usrp.py`:
    # i = np.right_shift(np.left_shift(data, 16), 16).astype('>f4') / 32767
    
    # The USRP will give us complex64 (float).
    # We should normalize and pack them the way `mockup_usrp` does.
    
    # recv_buffer is (1, N) complex64
    raw_samples = recv_buffer[0]
    
    i_rx = np.real(raw_samples)
    q_rx = np.imag(raw_samples)
    
    # Conversion to float32 bytes for TCP
    # Since we are already float, we just need to ensure endianness if needed
    # The mockup uses '>f4' (Big Endian Float)
    
    i_rx = i_rx.astype('>f4')
    q_rx = q_rx.astype('>f4')
    
    iq_concat = np.concatenate((i_rx, q_rx), axis=-1).astype('>f4')
    
    # Pack length + data
    send_payload = len(iq_concat).to_bytes(length=METADATA_BYTES, byteorder='big') + iq_concat.tobytes()
    
    client_socket.sendAll(send_payload)
    print("Forwarded received samples to client.")

def main():
    # 1. Connect to Client (acting as server in this context, similar to mockup)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((CLIENT_ADDR, CLIENT_PORT))
        print(f"Connected to client at {CLIENT_ADDR}:{CLIENT_PORT}")
    except ConnectionRefusedError:
        print(f"Could not connect to {CLIENT_ADDR}:{CLIENT_PORT}. Is client.py running?")
        sys.exit(1)

    # 2. Initialize USRP
    usrp = setup_usrp()
    
    # Set up Streamers
    st_args = uhd.usrp.StreamArgs("fc32", "sc16") # Host format float32, OTA format complex int16
    st_args.channels = [0]
    
    tx_streamer = usrp.get_tx_stream(st_args)
    rx_streamer = usrp.get_rx_stream(st_args)
    
    # Calculate buffer sizes
    # We expect to receive 4 * SAMPLE_SIZE bytes from TCP
    # SAMPLE_SIZE comes from 'pilot.py'
    # TCP Data Structure:
    # It sends: constellations (int32 array packed)
    
    print("USRP Driver Ready. Waiting for data...")
    
    try:
        while True:
            # A. Receive from TCP
            # This matches mockup_usrp behavior: receive_constellation_tcp
            # The client sends "constellations.tobytes()"
            # constellations are int32 (packed I/Q)
            
            raw_tcp_data = receive_constellation_tcp(sock, total_bytes=4 * SAMPLE_SIZE)
            if not raw_tcp_data:
                print("Disconnected from client.")
                break
                
            print("Received data from client.")
            
            # B. Parse and Convert for USRP Tx
            data_np = np.frombuffer(raw_tcp_data, dtype=np.int32).byteswap() # Big execution to Native?
            # Unpack packed int32 to floats
            # Logic from mockup_usrp:
            i = np.right_shift(np.left_shift(data_np, 16), 16).astype(np.float32) / 32767.0
            q = np.right_shift(data_np, 16).astype(np.float32) / 32767.0
            
            # Combine to complex64
            tx_complex = i + 1j * q
            tx_complex = tx_complex.astype(np.complex64)
            tx_complex = tx_complex.reshape(1, -1)
            
            # C. Transmit on USRP
            # Send burst
            meta_tx = uhd.types.TXMetadata()
            meta_tx.has_time_spec = False
            meta_tx.start_of_burst = True
            meta_tx.end_of_burst = True # Assuming one shot per image chunk
            
            num_tx = tx_streamer.send(tx_complex, meta_tx)
            print(f"Transmitted {num_tx} samples to USRP.")
            
            # D. Receive from USRP (Loopback/Echo/Channel response)
            # We assume the Rx captures the Tx (if loopback cable) or OTA signal
            # We assume we capture roughly the same number of samples
            # plus some delay?
            
            # For simplicity, we capture slightly more or same
            num_rx_req = tx_complex.shape[1] + 1024 # Add some buffer for delay/multipath
            
            # We might need to handle the time gap between Tx and Rx if they are not synchronized
            # or if we are waiting for a reflected signal.
            # But the mockup immediately returns noise+signal.
            
            # Setup Rx Stream
            meta_rx = uhd.types.RXMetadata()
            rx_buffer = np.zeros((1, num_rx_req), dtype=np.complex64)
            
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
            stream_cmd.num_samps = num_rx_req
            stream_cmd.stream_now = True
            rx_streamer.issue_stream_cmd(stream_cmd)
            
            num_rx = rx_streamer.recv(rx_buffer, meta_rx)
            print(f"Received {num_rx} samples from USRP.")
            
            if meta_rx.error_code != uhd.types.RXMetadataErrorCode.none:
                print(f"Rx Error: {meta_rx.strerror()}")
                
            # E. Send back to TCP client
            # The client expects:
            # len (4 bytes) + i_array (float32 big endian) + q_array (float32 big endian)
            
            rx_flat = rx_buffer[0, :num_rx] # Valid samples
            
            # Separate I and Q
            i_ret = np.real(rx_flat).astype('>f4')
            q_ret = np.imag(rx_flat).astype('>f4')
            
            # Concatenate [[i...], [q...]] NOT interleaved?
            # Mockup: iq_concat = np.concatenate((i, q), axis=-1) -> this is actually APPENDING q after i
            # Because i and q are 1D arrays.
            # So the format is I_BLOCK followed by Q_BLOCK.
            
            iq_concat = np.concatenate((i_ret, q_ret), axis=0).astype('>f4')
            
            length_bytes = len(iq_concat).to_bytes(length=METADATA_BYTES, byteorder='big')
            payload = length_bytes + iq_concat.tobytes()
            
            sock.sendall(payload)
            print("Sent Rx data back to client.")
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        sock.close()
        print("Closed connection.")

if __name__ == "__main__":
    main()
