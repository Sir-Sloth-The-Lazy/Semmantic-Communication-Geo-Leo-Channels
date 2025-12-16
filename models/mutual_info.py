
import tensorflow as tf

class Mine(tf.keras.Model):
    def __init__(self, in_dim=2, hidden_size=10):
        super(Mine, self).__init__()
        
        # Reference uses Random Normal(0.0, 0.02) for weights and Zero for bias
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        
        self.dense1 = tf.keras.layers.Dense(
            hidden_size, 
            kernel_initializer=initializer, 
            bias_initializer='zeros'
        )
        self.dense2 = tf.keras.layers.Dense(
            hidden_size, 
            kernel_initializer=initializer, 
            bias_initializer='zeros'
        )
        self.dense3 = tf.keras.layers.Dense(
            1, 
            kernel_initializer=initializer, 
            bias_initializer='zeros'
        )

    def call(self, inputs):
        x = self.dense1(inputs)
        x = tf.nn.relu(x)
        x = self.dense2(x)
        x = tf.nn.relu(x)
        output = self.dense3(x)
        return output

def mutual_information(joint, marginal, mine_net):
    """
    Estimates the Mutual Information lower bound using the MINE network.
    
    Args:
        joint: Batch of samples from the joint distribution P(X,Z).
        marginal: Batch of samples from the marginal distribution P(X)P(Z).
        mine_net: The MINE neural network.
        
    Returns:
        mi_lb: Lower bound of mutual information.
        t: Output of network on joint samples.
        et: Exponential output of network on marginal samples.
    """
    t = mine_net(joint)
    et = tf.exp(mine_net(marginal))
    
    # MINE Lower Bound = E_J[T] - log(E_M[exp(T)])
    mi_lb = tf.reduce_mean(t) - tf.math.log(tf.reduce_mean(et))
    
    return mi_lb, t, et

def sample_batch(rec, noise):
    """
    Creates joint and marginal samples from signal and noise.
    
    Args:
        rec: Received signal (or first variable X).
        noise: Noise (or second variable Z).
        
    Returns:
        joint: Concatenated pairs (X, Z).
        marg: Concatenated shuffled pairs (X, Z').
    """
    # Reshape to column vectors (-1, 1)
    rec = tf.reshape(rec, shape=(-1, 1))
    noise = tf.reshape(noise, shape=(-1, 1))
    
    # Ensure inputs are float32
    rec = tf.cast(rec, tf.float32)
    noise = tf.cast(noise, tf.float32)
    
    n_samples = tf.shape(rec)[0]
    half_n = n_samples // 2
    
    # Split into two halves
    # We use slicing instead of split to handle potential odd sizes gracefully (dropping last)
    rec_sample1 = rec[:half_n]
    # rec_sample2 = rec[half_n:2*half_n] # Unused for joint/marg structure in ref
    
    noise_sample1 = noise[:half_n]
    noise_sample2 = noise[half_n:2*half_n]
    
    # Joint Distribution: (rec1, noise1) -> Correlated/Paired
    joint = tf.concat([rec_sample1, noise_sample1], axis=1)
    
    # Marginal Distribution: (rec1, noise2) -> Independent/Unpaired
    marg = tf.concat([rec_sample1, noise_sample2], axis=1)
    
    return joint, marg

def learn_mine(batch, mine_net, ma_et, ma_rate=0.01):
    """
    Training step for MINE.
    
    Args:
        batch: Tuple of (joint, marginal) tensors.
        mine_net: The MINE network.
        ma_et: Moving average of exp(T).
        ma_rate: Moving average rate.
        
    Returns:
        loss: The loss to minimize (negative MI).
        ma_et: Updated moving average.
        mi_lb: Current MI lower bound estimate.
    """
    joint, marginal = batch
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)
    
    # Update moving average
    ma_et = (1 - ma_rate) * ma_et + ma_rate * tf.reduce_mean(et)
    
    # Biased estimator gradient reduction (often more stable)
    # loss = -(tf.reduce_mean(t) - (1 / tf.reduce_mean(ma_et)) * tf.reduce_mean(et))
    
    # For simplicity and sticking to the MINE definitions usually used:
    loss = -mi_lb
    
    return loss, ma_et, mi_lb
