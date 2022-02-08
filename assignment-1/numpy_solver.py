import numpy as np

def calculate_quantile(sequence, phi: float):
    assert 0 <= phi <= 1, 'phi must be between 0 and 1'
    return np.quantile(sequence, phi)