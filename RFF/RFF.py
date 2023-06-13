""" API for application kernel-based EDMD using Random Fourier Features.

    This module provides methods related to the application of kernelized Extended
    Dynamic Mode Decomposition (kEDMD) [1] to time series data, using low-rank
    approximations based on Random Fourier Features (RFF) [2].
    Available functions include:

    M_RFF_Koopman:
                compute RFF feature matrices for Koopman operator approximation
    M_RFF_Generator:
                compute RFF feature matrices for Koopman generator approximation
    cv_koopman_rff:
                perform cross-validation to estimate VAMP score for the purpose of model
                validation [3], [4] when approximating the Koopman operator
    cv_generator_rff:
                perform cross-validation to estimate VAMP score for the purpose of model
                validation [3], [4] when approximating the Koopman generator
    spectral_analysis_rff_koopman:
                compute eigenvalues and eigenvectors of the Koopman operator using RFFs.
    spectral_analysis_rff_generator:
                compute eigenvalues and eigenvectors of the Koopman generator using RFFs.

    [1] Klus et al., Eigendecompositions of transfer operators in reproducing kernel
     Hilbert spaces, Journal of Nonlinear Science, 2020
    [2] Nüske and Klus, Efficient Approximation of Molecular Kinetics using Random
     Fourier Features, arxiv:2306.00849
    [3] Noé and Nüske, A variational approach to modeling slow processes in stochastic
     dynamical systems, SIAM Multiscale Modeling & Simulation, 2013
    [3] Wu and Noé, Variational approach for learning Markov processes from time series
     data, Journal of Nonlinear Science, 2020
 """


import numpy as np
from scipy.linalg import svd, eig, eigvals, eigh, norm, eigvalsh

from sklearn.model_selection import train_test_split

from KoopmanLib.Util.util import split_by_lag, whitening_transform, filter_ev


""" Functions to compute RFF matrices feature matrices: """
def M_RFF_Koopman(X, Y, Omega):
    """
        Compute RFF feature matrices for approximation of the Koopman operator at
        finite lag time.

        Parameters:
        --------
        X (d, m):           data array, m points in d-dimensional space.
        Y (d, m):           time-shifted data array, m points in d-dimensional space.
        Omega (d, p):       Fourier features, p points in d-dimensional frequency space

        Return:
        --------
        M (m, p):        evaluation of Fourier features all data point
        Mt (m, p):       evaluation of Fourier features all time-shifted data points
    """
    # Compute Feature matrices:
    M = np.exp(-1j * np.dot(X.T, Omega))
    Mt = np.exp(-1j * np.dot(Y.T, Omega))

    return M, Mt

def M_RFF_Generator(X, Omega, a=1.0, b=None, reversible=False):
    """
        Compute feature matrices for RFF approximation of the Koopman generator
        for an SDE.

        Parameters:
        --------
        X (d, m):           data array, m points in d-dimensional space.
        Omega (d, p):       feature array, p point in d-dimensional frequency space
        a:                  diffusion tensor at all data sites. a can be
                            float or (d, d, m).
        b:                  drift field at all data sites. b can be None or (d, m).
        reversible, bool:   use reversible formulation. In this case, argument b
                            will be ignored.

        Return:
        --------
        M (m, p):       evaluation of Fourier features all data point
        ML (m, p) or (p, p):
                        application of generator appplied to all Fourier features at all
                        data points. If reversible is True, ML is the contraction of the
                        gradients of all Fourier features with the diffusion field, which
                        is a pxp-matrix.
    """
    # Determine dimensions:
    d, m = X.shape
    _, p = Omega.shape
    # Compute Feature feature matrix:
    M = np.exp(-1j * np.dot(X.T, Omega))
    # Compute second-order contrbutions for non-reversible case:
    if not reversible:
        if isinstance(a, float):
            # Use simplified formula if diffusion is constant:
            ML = -0.5 * a * M * (norm(Omega, axis=0)**2)[None, :]
        else:
            # Use general formula otherwise:
            omega_out = np.einsum("ik, jk -> ijk" , Omega, Omega)
            ML = -0.5 * np.einsum("ijk, ijl-> kl", a, omega_out) * M
        # Add first-order terms if present:
        if b is not None:
            ML += -1j * np.dot(b.T, Omega) * M

    # Compute contractions of RFF gradients with diffusion for reversible case:
    else:
        if isinstance(a, float):
            ML = -0.5 * a * np.dot(Omega.T, Omega) * np.dot(M.conj().T, M)
        else:
            # First, form two p x p x m arrays:
            Om_a = np.einsum("iu, ijl, jv -> uvl", Omega, a, Omega)
            Mtens = np.einsum("liu, liv -> uvl", M[:, None, :].conj(), M[:, None, :])
            # Then, contract over the data index:
            ML = -0.5 * np.sum(Om_a * Mtens, axis=2)

    return M, ML

""" Wrapper functions for model validation: """
def cv_koopman_rff(X, Omega, lag, rtrain, ntest, nev, tol=0.0, eps=1e-4):
    """
    Score RFF model for the Koopman operator using repeated random shuffling of the data.
    NOTE: This function is based on the VAC / symmetric VAMP score, assuming a reversible
    system.

    Parameters:
    -----------
    X, array (d, m) or [list of arrays (d, m_i)]:
                    full data, m points in d-dimensional space
    Omega, (d, p):  random Fourier features in d-dimensional space
    lag, int:       lag time for Koopman estimation
    rtrain, float:  ratio of training vs test data
    ntest, int:     number of random shuffles
    nev, int:       number of eigenvalues for VAMP score
    tol, float:     svd cutoff
    eps, float:     discard all eigenvalue with real part greater than 1.0 + eps.

    Returns
    -------
    d, array(ntest, nev)
                    eigenvalues computed for each random shuffle of the data
    dtest, array(ntest,)
                    test score for each random shuffle of the data

    """
    # Separate and concatenate data into time-shifted pairs:
    X, Y = split_by_lag(X, lag)
    # Prepare output:
    d = np.zeros((ntest, nev), dtype=complex)
    dtest = np.zeros(ntest)
    # Compute test score for each random shuffle of the data:
    for ii in range(ntest):
        # Random shuffling of the data:
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X.T, Y.T, train_size=rtrain)
        # Evaluate RFF basis:
        M, Mt = M_RFF_Koopman(Xtrain.T, Ytrain.T, Omega)
        # Compute whitening transformation:
        L, VM = whitening_transform(M.conj().T, tol, rmin=nev, return_V=True, complex=True)
        # Compute reduced matrix
        R = VM.conj().T @ Mt @ L
        # Solve eigenvalue problem:
        di, Wi = eig(R)
        # Filter and sort eigenvalues and eigenvectors:
        di, Wi = filter_ev(di, Wi, eps2=1.0+eps)
        # Compute transformation to eigenvector space:
        Wi = L @ Wi[:, -nev:]
        # Extract eigenvalues:
        d[ii, :] = di[-nev:]
        # Score the data:
        dtest[ii] = _score_test_data_koopman(Xtest.T, Ytest.T, Omega, Wi)

    return d, dtest

def cv_generator_rff(X, Omega, a, rtrain, ntest, nev, tol=0.0):
    """
    Score RFF model for the Koopman generator using repeated random shuffling of the data.
    NOTE: This function is based on the VAC / symmetric VAMP score, and therefore uses the
    reversible formulation for generator EDMD.

    Parameters:
    -----------
    X, array (d, m) or list of arrays(d, m_i)
                    full data, m points in d-dimensional space
    Omega, (d, p):  random Fourier features in d-dimensional space
    a, float or array(d, d, m) or list of arrays(d, d, m_i):
                    diffusion tensor at all data points.
    rtrain, float:  ratio of training vs test data
    ntest, int:     number of random shuffles
    nev, int:       number of eigenvalues for VAMP score
    tol, float:     svd cutoff

    Returns
    -------

    """
    # Concatenate data if given as lists:
    if isinstance(X, list):
        X = np.hstack(X)
        if not isinstance(a, float):
            a = np.concatenate([ia for ia in a], axis=2)

    # Prepare output:
    d = np.zeros((ntest, nev), dtype=complex)
    dtest = np.zeros(ntest)
    # Compute test score for each random shuffle of the data:
    for ii in range(ntest):
        # Random shuffling of the data:
        if isinstance(a, float):
            Xtrain, Xtest = train_test_split(X.T, train_size=rtrain)
            atrain = a
            atest = a
        else:
            Xtrain, Xtest, atrain, atest = train_test_split(X.T, a.transpose([2, 0, 1]),
                                                        train_size=rtrain)
            atrain = atrain.transpose([1, 2, 0])
            atest = atest.transpose([1, 2, 0])
        # Evaluate RFF basis:
        M, ML = M_RFF_Generator(Xtrain.T, Omega, atrain, b=None, reversible=True)
        # Compute whitening transformation:
        L = whitening_transform(M.conj().T, tol, rmin=nev, return_V=False, complex=True)
        # Compute reduced matrix
        R = L.conj().T @ ML @ L
        # Solve eigenvalue problem:
        di, Wi = eigh(R)
        # Filter and sort eigenvalues and eigenvectors:
        di, Wi = filter_ev(di, Wi)
        # Compute transformation to eigenvector space:
        Wi = L @ Wi[:, -nev:]
        # Extract eigenvalues:
        d[ii, :] = di[-nev:]
        # Score the data:
        dtest[ii] = _score_test_data_generator(Xtest.T, Omega, atest, Wi)

    return d, dtest

""" Wrapper functions for spectral analysis of Koopman operator or generator: """

def spectral_analysis_rff_koopman(X, Omega, lag, nev, tol=0.0):
    """
    Compute spectral decomposition of the Koopman operator based on random Fourier features

    Parameters:
    -----------
    X, array (d, m) or [list of arrays (d, m_i)]:
                    full data, m points in d-dimensional space
    Omega, (d, p):  random Fourier features in d-dimensional space
    lag, int:       lag time for Koopman estimation
    nev, int:       number of eigenvalues to be retained
    tol, float:     svd cutoff

    Returns
    -------
    d, (nev,)       eigenvalues of the Koopman operator
    W, (p, nev)     corresponding eigenvectors
    M, (m-lag, p)   random Fourier feature matrix at all data points
    """
    # Separate and concatenate data into time-shifted pairs:
    X, Y = split_by_lag(X, lag)
    # Generate Fourier matrices:
    M, Mt = M_RFF_Koopman(X, Y, Omega)
    # Compute whitening transformation:
    L, VM = whitening_transform(M.conj().T, tol, rmin=nev, return_V=True, complex=True)
    # Compute reduced matrix
    R = VM.conj().T @ Mt @ L
    # Solve eigenvalue problem and sort eigenvalues in increasing order:
    di, Wi = eig(R)
    # Filter and sort eigenvalues and eigenvectors:
    di, Wi = filter_ev(di, Wi)
    # Compute transformation to eigenvector space:
    W = L @ Wi[:, -nev:]
    # Extract eigenvalues:
    di = di[-nev:]

    return di, W, M

# TODO: add functionality to estimate coarse-grained generator
def spectral_analysis_rff_generator(X, Omega, nev, a, b=None, tol=0.0, reversible=False):
    """
        Compute spectral decomposition of the Koopman generator
         based on random Fourier features. NOTE: currently this function
         can only be used to learn a coarse-grained generator in the reversible setting.

        Parameters:
        -----------
        X, array (d, m) or list of arrays(d, m_i)
                    full data, m points in d-dimensional space
        Omega, (d, p):  random Fourier features in d-dimensional space.
        nev, int:       number of eigenvalues to be retained
        a, float or array(d, d, m) or list of arrays(d, d, m_i):
                    diffusion tensor at all data points.
        b, array(d, m) or list of arrays(d, m_i), or None.
                    drift field at all data sites.
        tol, float:     svd cutoff
        reversible, bool:   use reversible formulation. In this case, argument b
                            will be ignored.

        Returns
        -------
        d, (nev,)       eigenvalues of the Koopman generator
        W, (p, nev)     corresponding eigenvectors
        M, (m, p)   random Fourier feature matrix at all data points
        """
    # Concatenate data if given as lists:
    if isinstance(X, list):
        X = np.hstack(X)
        if not isinstance(a, float):
            a = np.concatenate([ia for ia in a], axis=2)
        if b is not None:
            b = np.concatenate([ib for ib in b], axis=1)

    # Check if b is given despite reversible flag being active:
    if reversible:
        b = None
    # Generate Fourier matrices:
    M, ML = M_RFF_Generator(X, Omega, a, b, reversible=reversible)
    # Compute whitening transformation:
    L, VM = whitening_transform(M.conj().T, tol, rmin=nev, return_V=True, complex=True)
    # Non-reversible case:
    if not reversible:
        # Compute reduced matrix
        R = VM.conj().T @ ML @ L
        # Solve eigenvalue problem and sort eigenvalues in increasing order:
        di, Wi = eig(R)
    else:
        # Compute reduced matrix
        R = L.conj().T @ ML @ L
        # Solve eigenvalue problem:
        di, Wi = eigh(R)
    # Filter and sort eigenvalues and eigenvectors:
    di, Wi = filter_ev(di, Wi)
    # Compute transformation to eigenvector space:
    W = L @ Wi[:, -nev:]
    # Extract eigenvalues:
    di = di[-nev:]

    return di, W, M


""" Private functions """

def _score_test_data_generator(Xtest, Omega, atest, L):
    """ Compute VAMP-score for pre-selected subspace of RFF basis. This function is intended for
        hyper-parameter tuning.

        Parameters:
        -----------
        Xtest, (d, m):  hold-out data used for scoring
        Omega, (d, p):  random Fourier features (should be the same as for training)
        atest, float or (d, d, m)
                        diffusion field at hold-out data sites
        L (p, r):       linear transformation to r-dimensional subspace

        Returns:
        -------
        dvamp, float:   test score on the data.
    """
    # Evaluate RFF basis:
    M, ML = M_RFF_Generator(Xtest, Omega, atest, b=None, reversible=True)
    # Compute reduced mass matrix:
    M1 = M @ L
    # Orthonormalize by SVD:
    U0, s0, W0 = svd(M1.conj().T, full_matrices=False)
    L0 = L @ (U0 * (s0 ** (-1))[None, :])
    _, r = L0.shape
    # Compute reduced matrix:
    R = L0.conj().T @ ML @ L0
    # Compute eigenvalues:
    dr = eigvalsh(R)

    return np.sum(dr)

def _score_test_data_koopman(Xtest, Ytest, Omega, L):
    """ Compute VAMP-score for pre-selected subspace of RFF basis. This function is intended for
        hyper-parameter tuning.

        Parameters:
        -----------
        Xtest, (d, m):  hold-out data used for scoring
        Ytest, (d, m):  time-shifted data used for scoring
        Omega, (d, p):  random Fourier features (should be the same as for training)
        L (p, r):       linear transformation to r-dimensional subspace

        Returns:
        -------
        dvamp, float:   test score on the data.
    """
    # Evaluate RFF basis:
    M, Mt = M_RFF_Koopman(Xtest, Ytest, Omega)
    # Compute reduced mass matrix:
    M1 = M @ L
    # Orthonormalize by SVD:
    U0, s0, W0 = svd(M1.conj().T, full_matrices=False)
    L0 = L @ (U0 * (s0 ** (-1))[None, :])
    W0 = W0.conj().T
    _, r = L0.shape
    # Compute reduced matrix:
    R = W0.conj().T @ Mt @ L0
    # Compute eigenvalues:
    dr = eigvals(R)

    return np.sum(np.real(dr))