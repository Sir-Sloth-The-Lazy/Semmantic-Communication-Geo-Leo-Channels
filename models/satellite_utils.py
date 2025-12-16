
import tensorflow as tf
import numpy as np
import math

# --- DATA FROM FONTÁN PAPER TABLES (Ported from codebase) ---
P_MATRICES = {
    'urban': {
        40: np.array([[0.8628, 0.0737, 0.0635],
                      [0.1247, 0.8214, 0.0539],
                      [0.0648, 0.0546, 0.8806]]),
        60: np.array([[0.8681, 0.0952, 0.0367],
                      [0.1300, 0.8429, 0.0271],
                      [0.0701, 0.0761, 0.8538]]),
        80: np.array([[0.8870, 0.0562, 0.0568],
                      [0.1489, 0.8039, 0.0472],
                      [0.0890, 0.0371, 0.8739]])
    },
    'suburban': {
        40: np.array([[0.8177, 0.1715, 0.0108],
                      [0.1544, 0.7997, 0.0459],
                      [0.1400, 0.1314, 0.7167]])
    },
    'open': {
        40: np.array([[0.9530, 0.0431, 0.0039],
                      [0.0515, 0.9347, 0.0138],
                      [0.0334, 0.0238, 0.9428]]),
        60: np.array([[0.9643, 0.0255, 0.0102],
                      [0.0628, 0.9171, 0.0201],
                      [0.0447, 0.0062, 0.9491]]),
        80: np.array([[0.9307, 0.0590, 0.0103],
                      [0.0292, 0.9506, 0.0202],
                      [0.0111, 0.0397, 0.9492]])
    }
}

LOO_PARAMS = {
    'urban': {
        40: { 
            1: {'alpha': -0.3, 'psi': 0.73, 'MP': -15.9}, # State 1
            2: {'alpha': -8.0, 'psi': 4.5,  'MP': -19.2}, # State 2
            3: {'alpha': -24.4,'psi': 4.5,  'MP': -19.0}  # State 3
        },
        60: { 
            1: {'alpha': -0.35, 'psi': 0.26, 'MP': -16.0},
            2: {'alpha': -6.3,  'psi': 1.4,  'MP': -13.0},
            3: {'alpha': -15.2, 'psi': 5.0,  'MP': -24.8}
        },
        80: { 
            1: {'alpha': -0.25, 'psi': 0.87, 'MP': -21.7},
            2: {'alpha': -6.6,  'psi': 2.3,  'MP': -13.0},
            3: {'alpha': -11.0, 'psi': 8.75, 'MP': -24.2}
        }
    },
    'suburban': {
        40: { 
            1: {'alpha': -1.0, 'psi': 0.5,  'MP': -13.0},
            2: {'alpha': -3.7, 'psi': 0.98, 'MP': -12.2},
            3: {'alpha': -15.0,'psi': 5.9,  'MP': -13.0}
        },
        60: { 
            1: {'alpha': -0.3, 'psi': 0.91, 'MP': -15.7},
            2: {'alpha': -2.0, 'psi': 0.5,  'MP': -13.0},
            3: {'alpha': -3.8, 'psi': 0.34, 'MP': -13.2}
        },
        80: { 
            1: {'alpha': -0.4, 'psi': 0.58, 'MP': -13.7},
            2: {'alpha': -2.5, 'psi': 0.2,  'MP': -16.0},
            3: {'alpha': -4.25,'psi': 3.0,  'MP': -25.0}
        }
    },
    'open': {
        40: { 
            1: {'alpha': 0.1,  'psi': 0.37, 'MP': -22.0},
            2: {'alpha': -1.0, 'psi': 0.5,  'MP': -22.0},
            3: {'alpha': -2.25,'psi': 0.13, 'MP': -21.2}
        },
        60: { 
            1: {'alpha': 0.0,  'psi': 0.12, 'MP': -24.9},
            2: {'alpha': -0.7, 'psi': 0.12, 'MP': -26.1},
            3: {'alpha': -1.4, 'psi': 0.25, 'MP': -23.1}
        },
        70: { 
            1: {'alpha': -0.1, 'psi': 0.25, 'MP': -22.5},
            2: {'alpha': -0.5, 'psi': 0.28, 'MP': -24.5},
            3: {'alpha': -0.75,'psi': 0.37, 'MP': -23.24}
        },
        80: { 
            1: {'alpha': 0.1,  'psi': 0.16, 'MP': -22.4},
            2: {'alpha': -0.4, 'psi': 0.15, 'MP': -23.5},
            3: {'alpha': -0.72,'psi': 0.27, 'MP': -22.0}
        }
    }
}

def slant_path_distance(el, h):
    """
    Calculates the slant path distance to a satellite using TensorFlow operations.
    
    Args:
        el (float or tf.Tensor): Elevation angle in degrees.
        h (float or tf.Tensor): Satellite altitude in km.

    Returns:
        tf.Tensor: Slant path distance in km.
    """
    # Ensure inputs are tensors and float32
    el = tf.cast(el, tf.float32)
    h = tf.cast(h, tf.float32)
    
    R_e = 6371.0  # Mean radius of the Earth in km
    
    # Convert elevation to radians: degrees * pi / 180
    el_rad = el * (math.pi / 180.0)

    # Use the Law of Cosines / Quadratic Formula derived for Slant range
    # d^2 + (2*Re*sin(el))*d + (Re^2 - (Re+h)^2) = 0
    # a*d^2 + b*d + c = 0
    
    a = 1.0
    b = 2.0 * R_e * tf.math.sin(el_rad)
    c = (R_e**2) - ((R_e + h)**2)
    
    discriminant = (b**2) - (4.0 * a * c)
    
    # We assume discriminant >= 0 for valid geometries
    # Using tf.math.sqrt
    distance_km = (-b + tf.math.sqrt(discriminant)) / (2.0 * a)

    return distance_km

def free_space_path_loss(d, f):
    """
    Calculates the Free Space Path Loss (FSPL) in dB using TensorFlow.

    Args:
        d (float or tf.Tensor): Distance in km.
        f (float or tf.Tensor): Frequency in GHz.

    Returns:
        tf.Tensor: FSPL in dB.
    """
    d = tf.cast(d, tf.float32)
    f = tf.cast(f, tf.float32)
    
    # Safe log10: log(x) / log(10)
    def log10(x):
        return tf.math.log(x) / tf.math.log(10.0)

    # FSPL (dB) = 20 * log10(d) + 20 * log10(f) + 92.45
    # (where d is in km and f is in GHz)
    
    # Using tf.maximum to avoid log(0) if d is 0 (though unlikely in sat scenarios)
    d_safe = tf.maximum(d, 1e-6)
    f_safe = tf.maximum(f, 1e-6)

    fspl_db = 20.0 * log10(d_safe) + 20.0 * log10(f_safe) + 92.45
    return fspl_db

def atmospheric_loss(lat, lon, f, el):
    """
    Placeholder for Atmospheric Loss calculation.
    
    Ideally, this would use the ITU-R P.618 or similar models to calculate 
    attenuation due to gases, clouds, rain, and scintillation using 
    coordinates (lat, lon), frequency (f), and elevation (el).
    
    For now, avoiding external dependencies like 'itur'.
    """
    return 0.0
