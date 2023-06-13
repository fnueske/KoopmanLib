""" API for application of EDMD to time series data.

    This module provides methods related to the application of Extended Dynamic Mode
    Decomposition (EDMD) [1] to time series data. Available functions include:

    cv_vamp_score:
                perform cross-validation to estimate VAMP score for the purpose of model
                validation [2], [3]
    spectral_analysis_edmd:
                compute eigenvalues and eigenvectors of EDMD matrix
    evaluate_basis:
                evaluate basis functions on time series data.


    [1] Williams et al., A data–driven approximation of the koopman operator:
     Extending dynamic mode decomposition, Journal of Nonlinear Science, 2015
    [2] Noé and Nüske, A variational approach to modeling slow processes in stochastic
     dynamical systems, SIAM Multiscale Modeling & Simulation, 2013
    [3] Wu and Noé, Variational approach for learning Markov processes from time series
     data, Journal of Nonlinear Science, 2020
 """

import numpy as np
import sympy

from scipy.linalg import svd, eig, eigvals
from sklearn.model_selection import train_test_split

from KoopmanLib.Util.util import filter_ev, Sym2numeric, whitening_transform

""" Methods for cross-validation of different model classes using VAMP score """
def cv_vamp_score(Xfull, phi, lag, rtrain, ntest, nev, tol=0.0, eps_ev=1e-3, sym_var=None):
    """
        Score RFF Koopman model using repeated random shuffling of the data

    Parameters:
    -----------
    Xfull, array (d, m) or [list of arrays (d, m_i)]:
                    full data, m points in d-dimensional space
    phi:            definition of the dictionary. This can either be a callable
                    function that converts a d x m-dimensional input to an n x p
                    dimensional functional time series, or a list of symbolic functions
                    if d variables. In the latter case, the symbolic variables representing
                    state space also need to be passed to this function.
    lag, int:       lag time for Koopman estimation
    rtrain, float:  ratio of training vs test data
    ntest, int:     number of random shuffles
    nev, int:       number of eigenvalues for VAMP score
    tol, float:     svd cutoff
    eps_ev, float:  discard all eigenvalue greater than 1.0 + eps_ev
    sym_var,        listof length d  list of symbolic variables for functions in phi.
                            Only needed if phi consists of symbolic functions.

    Returns
    -------

    """
    # TODO: Use VAMP score for non-reversible systems
    # TODO: Implement reversible formulation with symmetrization

    # Separate data into time-shifted pairs:
    if isinstance(Xfull, list):
        X = np.zeros((Xfull[0].shape[0], 0))
        Y = np.zeros((Xfull[0].shape[0], 0))
        for Xii in Xfull:
            X = np.concatenate((X, Xii[:, :-lag]), axis=1)
            Y = np.concatenate((Y, Xii[:, lag:]), axis=1)
    else:
        X = Xfull[:, :-lag]
        Y = Xfull[:, lag:]
    # Prepare output:
    d = np.zeros((ntest, nev), dtype=complex)
    dtest = np.zeros(ntest)
    for ii in range(ntest):
        # Random shuffling of the data:
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X.T, Y.T, train_size=rtrain)
        # Evaluate the basis:
        PhiX, PhiY = evaluate_basis(Xtrain.T, Ytrain.T, phi, sym_var=sym_var)
        # Compute whitening transformation:
        L, V = whitening_transform(PhiX, tol, return_V=True)
        # Compute reduced matrix:
        R = V.T @ PhiY.T @ L
        # Diagonalize:
        di, Wi = eig(R)
        # Sort and filter eigenvalues:
        di, Wi = filter_ev(di, Wi, eps2=1.0+eps_ev)
        # Compute transformation to eigenvector space:
        Wi = L @ Wi[:, -nev:]
        # Extract eigenvalues:
        d[ii, :] = di[-nev:]
        # Score the data:
        dtest[ii] = _score_test_data_koopman(Xtest.T, Ytest.T, phi, Wi, sym_var=sym_var)

    return d, dtest


""" Wrapper for spectral decomposition based on whitening transform """

def spectral_analysis_edmd(Xfull, phi, lag, nev, tol=0.0, eps_ev=1e-3, sym_var=None):
    """
    Perform spectral decomposition of the Koopman operator based on finite-
    dimensional basis set.

    Parameters:
    -----------
    Xfull, array (d, m) or [list of arrays (d, m_i)]:
                    full data, m points in d-dimensional space
    phi:            definition of the dictionary. This can either be a callable
                    function that converts a d x m-dimensional input to an n x p
                    dimensional functional time series, or a list of symbolic functions
                    of d variables. In the latter case, the symbolic variables representing
                    state space also need to be passed to this function.
    lag, int:       lag time for Koopman estimation
    nev, int:       number of eigenvalues to be retained
    tol, float:     relative svd cutoff for whitening transformation
    eps_ev, float:  discard all eigenvalue greater than 1.0 + eps_ev
    sym_var,        listof length d  list of symbolic variables for functions in phi.
                            Only needed if phi consists of symbolic functions.

    Returns
    -------
    d, (nev,)       eigenvalues of the Koopman operator
    W, (p, nev)     corresponding eigenvectors
    L, (n,rp)       whitening transformation matrix
    """
    # Separate data into time-shifted pairs:
    if isinstance(Xfull, list):
        X = np.zeros((Xfull[0].shape[0], 0))
        Y = np.zeros((Xfull[0].shape[0], 0))
        for Xii in Xfull:
            X = np.concatenate((X, Xii[:, :-lag]), axis=1)
            Y = np.concatenate((Y, Xii[:, lag:]), axis=1)
    else:
        X = Xfull[:, :-lag]
        Y = Xfull[:, lag:]
    # Evaluate the basis:
    PhiX, PhiY = evaluate_basis(X, Y, phi, sym_var=sym_var)
    # Compute whitening transformation:
    L, V = whitening_transform(PhiX, tol, return_V=True)
    # Compute reduced matrix:
    R = V.T @ PhiY.T @ L
    # Diagonalize:
    d, W = eig(R)
    # Sort and filter eigenvalues:
    d, W = filter_ev(d, W, eps2=1.0+eps_ev)
    d = d[-nev:]
    W = W[:, -nev:]
    # Transform eigenvectors to full basis:
    W = L @ W
    # Apply to data:
    Wdata = W.T @ PhiX

    return d, W, Wdata


""" Elementary methods to evaluate the basis and perform whitening transform """

def evaluate_basis(X, Y, phi, sym_var=None):
    """ Evaluate basis set (dictionary) on a given time series.

     Parameters:
     -----------
     X, array(d, m):        m data points in d-dimensional space, initial data points
     Y, array(d, m):        m data points in d-dimensional space, terminal data points
                            or shifted time series.
     phi:                   definition of the dictionary. This can either be a callable
                            function that converts a d x m-dimensional input to an n x p
                            dimensional functional time series, or a list of symbolic functions
                            if d variables. In the latter case, the symbolic variables representing
                            state space also need to be passed to this function.
     sym_var, listof length d  list of symbolic variables for functions in phi.
                            Only needed if phi consists of symbolic functions.

    Returns:
    --------
    PhiX, array(n, m-lag)   Evaluation of the basis set on the first m-lag data points.
    PhiY, array(n, m-lag)   Evaluation of the basis set on the last m-lag data points.

    """
    # Determine shapes:
    d, m = X.shape
    # Convert symbolic functions to callables if needed:
    if sym_var is not None:
        phi = Sym2numeric(phi, sym_var, ndiff=0)
    # Evaluate all basis functions at all data sites:
    PhiX = phi(X)
    PhiY = phi(Y)

    return PhiX, PhiY

""" Private Functions """

def _score_test_data_koopman(Xtest, Ytest, phi, L, sym_var=None):
    """
    Compute VAMP-score for pre-selected subspace of EDMD dictionary. This function is
     intended for hyper-parameter tuning.
     NOTE: The VAMP score is only meaningful for reversible dynamical systens.


        Parameters:
        -----------
        Xtest, (d, m):  data used for scoring
        Ytest, (d, m):  time-shifted data used for scoring
        phi:            definition of the dictionary. This can either be a callable
                    function that converts a d x m-dimensional input to an n x p
                    dimensional functional time series, or a list of symbolic functions
                    if d variables. In the latter case, the symbolic variables representing
                    state space also need to be passed to this function.
        L (n, r):   linear transformation to r-dimensional subspace
        sym_var,    listof length d  list of symbolic variables for functions in phi.
                            Only needed if phi consists of symbolic functions.

        Returns:
        -------
        dvamp, float:   test score on the data.
    """
    # Evaluate the basis:
    PhiX, PhiY = evaluate_basis(Xtest, Ytest, phi, sym_var=sym_var)
    # Compute reduced mass matrix:
    M1 = L.T @ PhiX
    # Orthonormalize by SVD:
    U0, s0, W0 = svd(M1, full_matrices=False)
    L0 = L @ (U0 * (s0 ** (-1))[None, :])
    _, r = L0.shape
    # Compute reduced matrix:
    R = W0 @ PhiY.T @ L0
    # Compute eigenvalues:
    dr = eigvals(R)

    return np.sum(np.real(dr))
