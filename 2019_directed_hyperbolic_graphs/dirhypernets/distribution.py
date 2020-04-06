'''
Functions to calculate different types of degree distribution:

- non-zero frequency degree distribution
- log-binning of degree distribution
- cumulative degree distribution (with linear/logarithmic bins)
'''

import numpy as np
import pandas as pd

def degree_distribution(degrees):
    '''
    Non-zero frequency degree distribution.
    
    Parameters
    ----------
    degrees : array like
        list of network/networks degrees
    
    Returns:
    --------
    numpy array, numpy array
        degree distribution
    
    '''
    rho = np.bincount(degrees).astype(float)/len(degrees)
    k = np.arange(len(degrees))[np.nonzero(rho)]
    rho_k = rho[np.nonzero(rho)]
    return k, rho_k

def log_binning(degrees, num):
    '''
    Log binning of degree distribution.
    
    Parameters
    ----------
    degrees : array like
        list of network/networks degrees
    num : integer
        bins count
    
    Returns:
    --------
    numpy array, numpy array
        degree distribution
    
    '''
    df = pd.DataFrame({
        'degree' : np.arange(max(degrees)+1)[max(1, min(degrees)):],
        'freq' : np.bincount(degrees).astype(float)[max(1, min(degrees)):]/len(degrees)
    })
    df['bin'] = np.digitize(df.degree, bins=np.geomspace(df.degree.min(), df.degree.max(), num=num, endpoint=False))
    x = df.groupby('bin').mean().degree
    y = df.groupby('bin').mean().freq
    return x, y

def cumulative_distribution(degrees, n_bins):
    '''
    Cumulative degree distribution.
    
    Parameters
    ----------
    degrees : array like
        list of network/networks degrees
    n_bins : integer
        bins count
    
    Returns:
    --------
    numpy array, numpy array
        degree distribution
    
    '''
    right_prop = np.vectorize(lambda split, x=degrees: np.sum([i > split for i in degrees]).astype(float)/len(degrees))
    x = np.linspace(min(degrees), max(degrees)+1, n_bins, endpoint=True)
    y = right_prop(x)
    return x, y

def log_cumulative_distribution(degrees, n_bins):
    '''
    Logarithmic cumulative degree distribution.
    
    Parameters
    ----------
    degrees : array like
        list of network/networks degrees
    n_bins : integer
        bins count
    
    Returns:
    --------
    numpy array, numpy array
        degree distribution
    
    '''
    right_prop = np.vectorize(lambda split, x=degrees: np.sum([i > split for i in degrees]).astype(float)/len(degrees))
    x = np.geomspace(max(1, min(degrees)), max(degrees)+1, n_bins, endpoint=True)
    y = right_prop(x)
    return x, y