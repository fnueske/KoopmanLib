""" This module provides utility functions for approximations based on Random Fourier features
 (RFF). Currently availble functions include:

 sample_rff_gaussian:
            generate samples from the spectral measure for the Gaussian kernel.
sample_rff_gauss_periodic:
            generate samples from the spectral measure for the periodic Gaussian kernel.
"""

import numpy as np
from scipy.special import iv

""" Functions to generate random Fourier features for Gaussians kernels """

def sample_rff_gaussian(d, p, sigma):
    """ Draw a sample from the spectral density for a Gaussian kernel.
    Parameters:
    -----------
    d, int:         dimension of the state space for Gaussian kernel
    p, int:         number of samples from spectral density
    sigma, float:   bandwidth of the Gaussian kernel

    Returns:
    --------
    Omega (d, p):   p samples drawn from d-dimensional spectral density

    """

    return (1.0 / sigma) * np.random.randn(d, p)

def sample_rff_gauss_periodic(d, p, L, sigma, kmax):
    """ Draw a sample from the spectral density for a periodic Gaussian kernel.
    Parameters:
    -----------
    d, int:         dimension of the state space for Gaussian kernel
    p, int:         number of samples from spectral density
    L, float:       half-period of the domain, i.e. the domain is [-L, L]^d.
    sigma, float:   bandwidth of the Gaussian kernel
    kmax, int:      maximal wavenumber; random frequencies are drawn from the
                    hyper-grid $omega_0 * [-kmax, kmax]^d$.

    Returns:
    --------
    Omega (d, p):   p samples drawn from d-dimensional spectral density

    """
    # Fundamental frequency:
    omega0 = np.pi / L
    # Admissible wavenumbers in one dimension:
    kvec = range(-kmax, kmax + 1)
    # Compute probabilities in one dimension:
    pvec = np.zeros((2 * kmax + 1,))
    for kk in range(-kmax, kmax + 1):
        pvec[kk + kmax] = iv(kk, sigma ** (-2)) / np.exp(sigma ** (-2))
    pvec /= np.sum(pvec)
    # Prepare output:
    Omega = np.zeros((d, p))
    # Generate random frequencies from product distribution:
    for jj in range(d):
        Omega[jj, :] = omega0 * np.random.choice(kvec, size=p, replace=True, p=pvec)

    return Omega