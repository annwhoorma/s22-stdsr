import numpy as np
import matplotlib.pyplot as plt

def generate_bimodal_normal(N:int=1000) -> np.ndarray:
    assert N % 2 == 0, 'size of the dataset must be an even number'
    mu, sigma = 50, 10
    mu2, sigma2 = 10, 20
    x1 = np.random.normal(mu, sigma, N//2)
    x2 = np.random.normal(mu2, sigma2, N//2)
    return np.concatenate([x1, x2])

def generate_normal(N:int=1000) -> np.ndarray:
    mu, sigma = 100, 50
    return np.random.normal(mu, sigma, N)

def generate_poisson(N:int=1000, lam:int=5) -> np.ndarray:
    return np.random.poisson(lam, N)

def generate_random(N:int=1000, start:int=0, end:int=10) -> np.ndarray:
    '''
    suitable only for one run!
    '''
    data = np.linspace(start, end, N)
    np.random.shuffle(data)
    probs = np.linspace(0, 1, N)
    probs = probs / np.sum(probs)
    np.random.shuffle(probs)
    data = np.random.choice(data, N, p=probs)
    return data
