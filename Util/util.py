""" This module is a collection of utilities that can be used while applying
    this toolbox. The following functionality is currently available:

    Methods:
    --------
    whitening transform:
            compute (reduced) transformation to an empirically orthonormal subspace
            for given basis set and time series.
    filter_ev:
            filter out eigenvalues which are too small or too large.
    split_by_lag:
            extract time-shifted pieces of a given list of trajectories, and concatenate
            them into two arrays of associated (x, y)-pairs.

    Classes:
    --------
    Sym2Numeric
            for a list of symbolic functions, this class creates an object that can be called
            to evaluate the functions and their derivatives at given data points.

"""

import numpy as np
import sympy

from scipy.linalg import svd

""" Function to compute whitening transformation """

def whitening_transform(PhiX, tol, rmin=0, return_V=False, complex=False):
    """ Compute whitening transformation of the mass matrix based on truncated
        singular value decomposition.

      Parameters:
     -----------
     PhiX, array(n, m):     functional (lifted) time series, i.e. evalutation of n
                            basis functions at m data points.
                            NOTE: In Koopman operator estimation, PhiX only comprise the
                            first m-t time steps, where t is the lag time.
     tol, float             relative truncation threshold for SVD
     rmin, int:             minimal rank to be retained when truncating singular values
     return_V, bool:        return matrix of right singular vectors of the data matrix

    Returns:
    --------
    L, array(n, r)          linear transformation to reduced basis obtained from SVD
    V, array(m, r)          reduced right singular vectors, if return_V==True.
    """
    # Compute SVD:
    U, s, V = svd(PhiX, full_matrices=False)
    # Determine truncation rank and shrink SVD components:
    ind = np.where(s / s[0] >= tol)[0]
    r = np.maximum(ind.shape[0], rmin)
    U = U[:, :r]
    s = s[:r]
    if complex:
        V = V[:r, :].conj().T
    else:
        V = V[:r, :].T
    # Compute whitening transformation:
    L = U * (s**(-1))[None, :]
    if return_V:
        return L, V
    else:
        return L

""" Functions to filter eigenvalues and eigenvectors """
def filter_ev(d, W, eps1=-np.infty, eps2=np.infty):
    """
    Sort and filter eigenvalues for the Koopman operator. Sorts eigenvalues
    in ascending order by magnitude of the real part. Also, remove all
    eigenvalues exceeding 1.0 + eps.

    Parameters:
    ----------
    d, array(n):        eigenvalues
    W, array(n, r)      asscoiated eigenvectors
    eps1, eps2, float   discard all eigenvalues with real part smaller than eps1
                        or greater than eps2

    Return:
    ------
    d, W,               sorted eigenvalues and eigenvectors
    """
    # Sort in ascending order by magnitude of the real part:
    ind = np.argsort(np.real(d))
    d = d[ind]
    W = W[:, ind]
    # Filter out spurious ev larger than one plus eps:
    ind = np.where(np.logical_and(np.real(d) > eps1, np.real(d) < eps2))[0]
    d = d[ind]
    W = W[:, ind]

    return d, W

""" Function to split given data into time-shifted parts: """
def split_by_lag(Xfull, lag):
    """
    Parameters:
    -----------
    Xfull, array (d, m) or [list of arrays (d, m_i)]:
                    full data, m points in d-dimensional space
    lag, int:       lag time for Koopman estimation

    Returns:
    --------
    X, Y (d, M):    time-shifted concatenated data arrays Each row of
                    Y is the time-shited data point for the same row in X.
    """
    if isinstance(Xfull, list):
        X = np.zeros((Xfull[0].shape[0], 0))
        Y = np.zeros((Xfull[0].shape[0], 0))
        for Xii in Xfull:
            X = np.concatenate((X, Xii[:, :-lag]), axis=1)
            Y = np.concatenate((Y, Xii[:, lag:]), axis=1)
    else:
        X = Xfull[:, :-lag]
        Y = Xfull[:, lag:]

    return X, Y

""" Abstract object to convert symbolic functions into callable functions """

class Sym2numeric:

    def __init__(self, psi_list, var_list, ndiff=0):
        """ This class allows to pass a list of symbolic functions and create an object that
        can be called to evaluate the basis at any given point. First and second order
        derivatives can also be evaluated.

        Parameters:
        -----------
        psi_list, list:         symbolic functions defining the basis set.
        var_list, list:         list of symbolic variables defining state space.
        ndiff, int:             Number of derivatives to be computed for all symbolic functions
        """
        # Get info:
        self.psi = psi_list
        self.var = var_list
        self.n = len(self.psi)
        self.d = len(self.var)
        self.ndiff = ndiff
        # Generate numerical functions:
        self.psi_eval = [sympy.lambdify(self.var, self.psi[ii], "numpy") for ii in range(self.n)]
        # Generate numerical functions for first and second order derivatives:
        if self.ndiff > 0:
            self.dpsi_eval = [[sympy.lambdify(self.var, self.psi[ii].diff(self.var[jj]), "numpy")
                           for jj in range(self.d)] for ii in range(self.n)]
        if self.ndiff > 1:
            self.ddpsi_eval = [[[sympy.lambdify(self.var, (self.psi[ii]).diff(self.var[kk]).diff(self.var[jj]), "numpy")
                               for kk in range(self.d)] for jj in range(self.d)] for ii in range(self.n)]

    def __call__(self, x):
        """ Evaluate the basis set along array of positions x

        Parameters:
        -----------
        x, ndarray(d, m):       array of m positions in d-dimensional space.

        Returns:
        --------
        psi_x, ndarray(n, m):   Evaluation of the basis set at all positions.
        """
        # Get info:
        m = x.shape[1]
        # Prepare output:
        psi_x = np.zeros((self.n, m))
        for ii in range(self.n):
            psi_x[ii, :] = self.psi_eval[ii](*[x[ll, :] for ll in range(self.d)])

        return psi_x

    def diff(self, x):
        """ Evaluate gradients of the basis set along array of positions x

        Parameters:
        -----------
        x, ndarray(d, m):       array of m positions in d-dimensional space.

        Returns:
        --------
        dpsi_x, ndarray(n, d, m):   Evaluation of the gradients of the basis set at all positions.

        """
        if self.ndiff < 1:
            raise AttributeError("First order derivatives not provided this instance of Sym2numeric")
        # Get info:
        m = x.shape[1]
        # Prepare output:
        dpsi_x = np.zeros((self.n, self.d, m))
        for ii in range(self.n):
            for jj in range(self.d):
                dpsi_x[ii, jj, :] = self.dpsi_eval[ii][jj](*[x[ll, :] for ll in range(self.d)])

        return dpsi_x

    def ddiff(self, x):
        """ Evaluate Hessian of the basis set along array of positions x

        Parameters:
        -----------
        x, ndarray(d, m):       array of m positions in d-dimensional space.

        Returns:
        --------
        ddpsi_x, ndarray(n, d, d, m):   Evaluation of the Hessian of the basis set at all positions.

        """
        if self.ndiff < 2:
            raise AttributeError("Second derivatives not provided this instance of Sym2numeric")
        # Get info:
        m = x.shape[1]
        # Prepare output:
        ddpsi_x = np.zeros((self.n, self.d, self.d, m))
        for ii in range(self.n):
            for jj in range(self.d):
                for kk in range(self.d):
                    ddpsi_x[ii, jj, kk, :] = self.ddpsi_eval[ii][jj][kk](*[x[ll, :] for ll in range(self.d)])

        return ddpsi_x