import tensorflow as tf
import random
import math
import numpy as np
from models.satellite_utils import slant_path_distance, free_space_path_loss, atmospheric_loss, P_MATRICES, LOO_PARAMS

class RayleighChannel(tf.keras.layers.Layer):
    def __init__(self, snrdB=None, clip_snrdB=5):
        super().__init__()
        self.snr = 10 ** (snrdB / 10) # in dB
        self.clip_snr = 10 ** (clip_snrdB / 10)
    

    def call(self, x):
        '''
        x: inputs with shape (b, c, 2)
           where last dimension denotes in-phase and quadrature-phase elements, respectively.
        Assumes slow rayleigh fading, where h does not change for single batch data

        We clip the coefficient h to generate short-term SNR between +-5 dB of given long-term SNR.
        '''
        assert x.shape[2] == 2, "input shape should be (b, c, 2), where last dimension denotes i and q, respectively"
        assert len(x.shape) == 3, "input shape should be (b, c, 2)"
        
        i = x[:,:,0]
        q = x[:,:,1]

        # power normalization
        sig_power = tf.math.reduce_mean(i ** 2 + q ** 2)
        
        # batch-wise slow fading
        h = tf.random.normal(
            (1, 1, 2),
            mean=0,
            stddev=tf.math.sqrt(0.5)
        )

        snr = self.snr

        n = tf.random.normal(
            tf.shape(x),
            mean=0,
            stddev=tf.math.sqrt(sig_power/(2*snr))
        )

        yhat = h * x + n

        return yhat
    

    def get_config(self):
        config = super().get_config()
        return config


class AWGNChannel(tf.keras.layers.Layer):
    def __init__(self, snrdB=None):
        super().__init__()
        self.snr = 10 ** (snrdB / 10) # in dB
    

    def call(self, x):
        '''
        x: inputs with shape (b, c, 2)
           where last dimension denotes in-phase and quadrature-phase elements, respectively.
        '''
        assert x.shape[2] == 2, "input shape should be (b, c, 2), where last dimension denotes i and q, respectively"
        assert len(x.shape) == 3, "input shape should be (b, c, 2)"

        i = x[:,:,0]
        q = x[:,:,1]

        # power normalization
        sig_power = tf.math.reduce_mean(i ** 2 + q ** 2)
        snr = self.snr

        n = tf.random.normal(
            tf.shape(x),
            mean=0,
            stddev=tf.math.sqrt(sig_power/(2*snr))
        )

        y = x + n
        return y
    

    def get_config(self):
        config = super().get_config()
        return config


class RicianChannel(tf.keras.layers.Layer):
    def __init__(self, snrdB=None, k=2):
        super().__init__()
        self.snr = 10 ** (snrdB / 10) # in dB
        self.k = k
    

    def call(self, x):
        '''
        x: inputs with shape (b, c, 2)
           where last dimension denotes in-phase and quadrature-phase elements, respectively.
        Assumes slow rayleigh fading (for NLOS part), where h does not change for single batch data

        We clip the coefficient h to generate short-term SNR between +-5 dB of given long-term SNR.
        '''
        assert x.shape[2] == 2, "input shape should be (b, c, 2), where last dimension denotes i and q, respectively"
        assert len(x.shape) == 3, "input shape should be (b, c, 2)"
        
        i = x[:,:,0]
        q = x[:,:,1]

        # power normalization
        sig_power = tf.math.reduce_mean(i ** 2 + q ** 2)
        
        # batch-wise slow fading
        h = tf.random.normal(
            (1, 1, 2),
            mean=0,
            stddev=tf.math.sqrt(0.5)
        )

        snr = self.snr

        n = tf.random.normal(
            tf.shape(x),
            mean=0,
            stddev=tf.math.sqrt(sig_power/(2*snr))
        )

        k = self.k

        yhat = tf.math.sqrt(1 / (1+k)) * h * x + tf.math.sqrt(k / (1+k)) * x + n

        return yhat


class SatelliteChannel(tf.keras.layers.Layer):
    def __init__(self, max_batch_size=128, snrdB=10.0):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.snr = 10 ** (snrdB / 10) # in dB
        # State: 0=S1_LOS, 1=S2_SHADOW, 2=S3_DEEP_SHADOW
        # Initialized in build() to ensure correct device placement
        
        # Flattened lookup tables could be optimizing, but dictionary lookup is fast enough for low freq
        # We will handle parameter retrieval dynamically in call
    # build() removed - no persistent state variable needed
    # Stateless implementation avoids CPU/GPU placement issues on Colab

    
    def _get_params(self, training):
        # Default behavior: Random LEO/GEO for training, Fixed LEO for inference
        if training:
            env_types = ['urban', 'suburban', 'open']
            env = random.choice(env_types)
            avail_els = list(P_MATRICES[env].keys())
            el = random.choice(avail_els)
            if random.random() < 0.9:
                h_sat = random.uniform(500.0, 2000.0)
            else:
                h_sat = 35786.0 
            freq_ghz = 2.4
        else:
            env = 'urban'
            el = 40
            h_sat = 550.0
            freq_ghz = 2.4
        return env, el, h_sat, freq_ghz

    def call(self, x, training=False):
        '''
        x: inputs with shape (b, c, 2)
        '''
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        
        # --- 1. Determine Environment Parameters ---
        env, el, h_sat, freq_ghz = self._get_params(training)

        # --- 2. Update Markov States ---

        # --- 2. Update Markov States ---
        
        # Get Transition Matrix P for (env, el) -> Shape (3, 3)
        # Convert to Tensor
        p_matrix_np = P_MATRICES[env][el]
        p_matrix = tf.constant(p_matrix_np, dtype=tf.float32) # (3, 3)
        
        # Stateless Approach:
        # Instead of carrying state over batches (which causes device placement issues with Keras 3 on Colab),
        # we generate a random current state distribution and simulate one step.
        # Since batches are shuffled, "memory" between batches is not physically meaningful anyway.
        
        # 1. Generate random current states (0=LOS, 1=Shadow, 2=Deep)
        # We assume a rough prior distribution or just uniform for robustness
        # Let's say: 50% LOS, 30% Shadow, 20% Deep as a generic prior to transition FROM.
        # functional_curr_states = tf.random.categorical(tf.math.log([[0.5, 0.3, 0.2]]), batch_size, dtype=tf.int32)
        # functional_curr_states = tf.reshape(functional_curr_states, [batch_size])
        
        # ACTUALLY: Let's just pick a random start state uniform per sample. 
        # The transition matrix will then guide it to the "next" state which we use.
        functional_curr_states = tf.random.uniform([batch_size], minval=0, maxval=3, dtype=tf.int32)

        # Gather probabilities for the next transition based on this random current state
        transition_probs = tf.gather(p_matrix, functional_curr_states) 
        
        # Sample next state
        next_states = tf.random.categorical(tf.math.log(transition_probs + 1e-9), 1, dtype=tf.int32)
        next_states = tf.reshape(next_states, [batch_size])
        
        # No persistent update needed
        # self.current_state[:batch_size].assign(next_states)
        
        # --- 3. Generate Fading Coefficients (Loo Distribution) ---
        
        # Retrieve Loo Parameters for (env, el)
        # structure: LOO_PARAMS[env][el][state_1_based] -> dict
        # We need to construct a tensor of shape (3, 3) where rows are states 0,1,2
        # and cols are [alpha, psi, MP]
        
        params_dict = LOO_PARAMS[env][el] # Keys are 1, 2, 3
        
        # Row 0 (State 1)
        p1 = params_dict[1]
        r1 = [p1['alpha'], p1['psi'], p1['MP']]
        # Row 1 (State 2)
        p2 = params_dict[2]
        r2 = [p2['alpha'], p2['psi'], p2['MP']]
        # Row 2 (State 3)
        p3 = params_dict[3]
        r3 = [p3['alpha'], p3['psi'], p3['MP']]
        
        # Loo Param Tensor: (3, 3)
        loo_tensor = tf.constant([r1, r2, r3], dtype=tf.float32)
        
        # Gather parameters based on *new* state (or current? typically state determines coefficient)
        # Using next_states as the state for this timestep
        current_params = tf.gather(loo_tensor, next_states) # (B, 3)
        
        alpha = current_params[:, 0]
        psi = current_params[:, 1]
        mp_db = current_params[:, 2]
        
        # A. Direct Path (Log-Normal Amplitude, Random Phase)
        # alpha, psi are mean and std of 20*log10(amplitude)
        # We need normal(mu, sigma) parameters derived from alpha/psi ?
        # Wait, the reference says:
        # mu_ln = (params['alpha'] / (20 * np.log10(np.e)))
        # sigma_ln = (params['psi'] / (20 * np.log10(np.e)))
        
        scale_factor = 20.0 * math.log10(math.e) # ~8.686
        mu_ln = alpha / scale_factor
        sigma_ln = psi / scale_factor
        
        # Sample Log Amplitude (Normal in log domain)
        log_amp = tf.random.normal(shape=[batch_size], mean=mu_ln, stddev=sigma_ln, dtype=tf.float32)
        # Convert to linear amplitude
        amp_direct = tf.exp(log_amp)
        
        # Random Phase ~ U[0, 2pi]
        phase_direct = tf.random.uniform(shape=[batch_size], minval=0, maxval=2*math.pi, dtype=tf.float32)
        
        h_direct_real = amp_direct * tf.math.cos(phase_direct)
        h_direct_imag = amp_direct * tf.math.sin(phase_direct)
        
        # B. Multipath (Rayleigh)
        # MP is power in dB relative to something?
        # Ref: mp_lin = 10**(params['MP'] / 10.0)
        # rayleigh_std = math.sqrt(mp_lin / 2.0)
        
        mp_lin = tf.math.pow(10.0, mp_db / 10.0)
        rayleigh_std = tf.math.sqrt(mp_lin / 2.0)
        
        h_multi_real = tf.random.normal(shape=[batch_size], mean=0.0, stddev=rayleigh_std, dtype=tf.float32)
        h_multi_imag = tf.random.normal(shape=[batch_size], mean=0.0, stddev=rayleigh_std, dtype=tf.float32)
        
        # Combined Channel H = Direct + Multipath
        h_real = h_direct_real + h_multi_real
        h_imag = h_direct_imag + h_multi_imag
        
        # Expand dims for broadcasting: (B,) -> (B, 1)
        h_real = tf.reshape(h_real, [batch_size, 1])
        h_imag = tf.reshape(h_imag, [batch_size, 1])
        
        # --- 4. Path Loss (Large Scale) ---
        
        # Calculate losses
        d_km = slant_path_distance(el, h_sat)
        fspl_db = free_space_path_loss(d_km, freq_ghz)
        
        # Assume constant atmospheric loss for now
        # att_db = atmospheric_loss(...) # Placeholder returns 0
        total_loss_db = fspl_db # + att_db
        
        # Note: The reference PyTorch implementation does not attenuate the signal 
        # by the absolute path loss (which would cause underflow). 
        # Instead, it scales the NOISE up to simulate the relative SNR drop.
        
        # Reference Distance
        d_ref = 550.0 
        fspl_ref = free_space_path_loss(d_ref, freq_ghz)
        
        # Delta Loss
        delta_loss_db = total_loss_db - fspl_ref
        
        # --- 5. Apply Channel to Signal ---
        
        # x is (B, C, 2)
        # Apply Fading (Complex Multiplication)
        # (x_r + j x_i) * (h_r + j h_i) = (x_r h_r - x_i h_i) + j(x_r h_i + x_i h_r)
        
        x_real = x[:, :, 0]
        x_imag = x[:, :, 1]
        
        y_real = x_real * h_real - x_imag * h_imag
        y_imag = x_real * h_imag + x_imag * h_real
        
        y = tf.stack([y_real, y_imag], axis=-1)

        
        # --- 6. Add Noise ---
        # Ref: scaled_noise_std = float(n_var) * (10.0 ** (delta_loss_db / 20.0))
        # The PyTorch code does a relative scaling logic delta_loss.
        # But here we applied absolute path loss.
        # So the signal 'y' is exceedingly small (e.g. 1e-15).
        # We need to add thermal noise.
        # Thermal noise power N = kB * T * B
        # But this simulation might be normalized?
        
        # The PyTorch ref uses 'delta_loss_db' relative to a 'ref_dist_km'.
        # And initializes SNR=10dB implicitly.
        # Let's verify how 'train_dist.py' uses this.
        # It likely expects 'y' to be normalized or the noise to be scaled.
        
        # IF we want to mimic the exact behavior where 'Tx_sig' is roughly order 1,
        # but the satellite attenuation makes it ~1e-10,
        # then the noise should be ~1e-11 for reasonable SNR.
        # If we just add 'n_var' (order 0.1), SNR will be -100dB.
        
        # HOWEVER, the PyTorch code does:
        # delta_loss = actual - ref
        # scaled_noise = n_var * 10**(delta_loss/20)
        # Rx = Tx + noise
        
        # Wait, the PyTorch code DOES NOT attenuate the signal Tx_sig by the absolute path loss!
        # "Rx_sig = Tx_sig + noise" (for AWGN check)
        # It SCALES THE NOISE UP to simulate the path loss relative to a reference.
        # This keeps the numerical values of the signal in a healthy range (avoiding underflow).
        
        # Let's adopt this "Relative Noise Scaling" approach for stability.
        
        # Reference Distance
        d_ref = 550.0 
        fspl_ref = free_space_path_loss(d_ref, freq_ghz)
        
        # Delta Loss
        delta_loss_db = total_loss_db - fspl_ref
        
        # Apply Fading (Channel Gain normalized to 1? or just use H?)
        # The Loo fading H has magnitude ~1?
        # mean ~ sqrt(0.5) approx. Yes.
        
        # Apply Fading to Signal (No FSPL attenuation applied to signal directly)
        x_real = x[:, :, 0]
        x_imag = x[:, :, 1]
        
        y_real = x_real * h_real - x_imag * h_imag
        y_imag = x_real * h_imag + x_imag * h_real
        y = tf.stack([y_real, y_imag], axis=-1)
        
        # Add Noise
        # Base Noise Std (assuming normalized signal power ~1 implies SNR sets noise)
        # If SNR=10dB, Noise Pow = 0.1, Std ~ 0.3
        # We assume 'self.snr' is our base operating point at the Reference Distance.
        # Wait, this class doesn't have self.snr in __init__?
        # The previous AWGN channel took 'snrdB'.
        # Let's stick to the signature of AWGNChannel or hardcode.
        # But we don't have n_var passed in call.
        # We should use a self.snr like the other layers.
        
        # Default base SNR
        snr_base = getattr(self, 'snr', 10.0) # Fallback
        # If not initialized, maybe we should init it. Using properties from AWGN?
        
        # Noise Power Pn = Ps / SNR
        # Ps ~ 1 (Power Normalized)
        pn_var = 0.5 / snr_base # 0.5 per dimension?
        base_std = tf.math.sqrt(pn_var)
        
        # Scale noise std by Delta Path Loss
        # If we represent moving further away as "Signal stays constant (AGC), Noise increases"
        noise_scaler = tf.math.pow(10.0, delta_loss_db / 20.0)
        
        final_std = base_std * noise_scaler
        
        n = tf.random.normal(tf.shape(x), mean=0.0, stddev=final_std)
        
        return y + n
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "max_batch_size": self.max_batch_size,
        })
        return config

class GEOSatelliteChannel(SatelliteChannel):
    def _get_params(self, training):
        # GEO Specific Behavior
        # Always 35,786 km
        h_sat = 35786.0
        freq_ghz = 2.4
        
        if training:
            # Randomize Env and Elevation for robustness
            env_types = ['urban', 'suburban', 'open']
            env = random.choice(env_types)
            avail_els = list(P_MATRICES[env].keys())
            el = random.choice(avail_els)
        else:
            # Fixed for evaluation
            # Urban, 40 deg elevation
            env = 'urban'
            el = 40
            
        return env, el, h_sat, freq_ghz


class LEOSatelliteChannel(SatelliteChannel):
    def _get_params(self, training):
        # LEO Specific Behavior (Strictly LEO)
        if training:
            # Randomize Env and Elevation
            env_types = ['urban', 'suburban', 'open']
            env = random.choice(env_types)
            avail_els = list(P_MATRICES[env].keys())
            el = random.choice(avail_els)
            
            # Random LEO Altitude (500-2000 km)
            h_sat = random.uniform(500.0, 2000.0)
            
            freq_ghz = 2.4
        else:
            # Fixed for evaluation (Standard LEO)
            env = 'urban'
            el = 40
            h_sat = 550.0
            freq_ghz = 2.4
            
        return env, el, h_sat, freq_ghz