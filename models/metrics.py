
import tensorflow as tf

def psnr(y_true, y_pred):
    """
    Peak Signal-to-Noise Ratio (PSNR).
    Matches implementation in analysis/psnr.py.
    NB: Assumes input range [0, 1].
    """
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim(y_true, y_pred):
    """
    Structural Similarity Index (SSIM).
    Matches implementation in analysis/ssim.py.
    NB: Assumes input range [0, 1].
    """
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def get_avg_cossim(x):
    '''
    Get average cosine similarity along spatial domain, except for self-similarity.
    Matches implementation in analysis/cossim.py.
    x: tensor with shape (B, H, W, C)
    '''
    # Ensure standard TF tensor
    b = tf.shape(x)[0]
    h = tf.shape(x)[1]
    w = tf.shape(x)[2]
    c = tf.shape(x)[3]
    
    # assert h == w, 'h should be equal to w' # Relax assertion for general use? 
    # analysis/cossim.py enforces it. We'll leave it out or handle logic carefully.
    
    x1 = tf.reshape(x, (-1, h*w, c))
    x2 = tf.reshape(x, (-1, h*w, c))

    cossim = tf.einsum('bic,bjc->bij', x1, x2)
    normalizer = tf.norm(x1, axis=-1, keepdims=True) * tf.reshape(tf.norm(x2, axis=-1), (-1, 1, h*w))
    cossim = cossim / (normalizer + 1e-9) # Add epsilon for safety

    # remove diagonal elements
    cossim = tf.linalg.set_diag(cossim, tf.zeros(cossim.shape[0:-1]))
    
    # Calculation from cossim.py
    # Avg over (B * (N^2 - N)) pairs
    N = tf.cast(h*w, tf.float32)
    B_float = tf.cast(b, tf.float32)
    denom = B_float * (N*N - N)
    
    avg_cossim = tf.reduce_sum(cossim) / tf.maximum(denom, 1.0)

    return avg_cossim
