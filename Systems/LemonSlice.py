import numpy as np

from scipy.integrate import quad, nquad

""" This module provides a class for the two-dimensional Lemon Slice potential [1],
    which is a model system for metastable stochastic dynamics. The potential is
    given by
     
     V(x, y) = cos(k*phi) + 1.0/np.cos(0.5*phi) + 10*(r-1)**2 + (1.0/r)

    where r, phi are the polar coordinates of x, y. The dynamics are given by the 
    stochastic differential equation

        dX_t = -\nabla V(X_t) dt + \sqrt{2\beta^{-1} } dW_t.
    
    The system class provides the following methods:
    
    Simulate:
            produce realizations of the system dynamics
    potential:
            evaluate the potential energy at given points
    gradient:
            evaluate the potential gradient at given points
    potential:
            evaluate the invariant density at given points
    
    [1] Bittracher et al, Transition manifolds of complex metastable systems.
     Journal of Nonlinear Science, 2018.
 """

class LemonSlice:
    """
        Stochastic dynamics in two-dimensional Lemon Slice potential:

        Parameters:
        -----------
        k, int:
            number of minima along the polar axis.
        beta, float:
            inverse temperature controlling noise intensity
    """

    def __init__(self, k, beta):
        self.k = k
        self.beta = beta

        # Compute partition function:
        dims = np.array([[-2.51, 2.5],
                         [-2.51, 2.5]])
        self.Z = nquad(lambda x, y: np.exp(-self.beta * self.potential(np.array([[x], [y]]))),
                       dims)[0]

        # Calculate normalization constants for effective dynamics:
        self.C1 = quad(lambda r: np.exp(-(10 * (r - 1) ** 2 + (1.0 / r))) * (1.0 / r), 0, 6.0)[0]
        self.C2 = quad(lambda r: np.exp(-(10 * (r - 1) ** 2 + (1.0 / r))) * r, 0, 6.0)[0]

    def Simulate(self, x0, m, dt):
        """
            Generate trajectory of stochastic dynamics in Lemon Slice potential, using Euler scheme.

        x0, nd-array(2):
            initial values for the simulation.
        m, int:
            number time steps to be returned (including initial value)
        dt, float:
            integration time step

        Returns:
        --------
        X, nd-array (2, m):
            simulation trajectory
        """
        # Initialize:
        X = np.zeros((2, m))
        X_old = x0[:, None]
        X[:, 0] = X_old[:, 0]
        # Run simulation:
        for t in range(1, m):
            # Euler-Maruyama update step:
            X_new = X_old - self.gradient(X_old) * dt + \
                    np.sqrt(2 * dt / self.beta) * np.random.randn(2, 1)
            #print(X_new)
            X[:, t] = X_new[:, 0]
            X_old = X_new

        return X

    def potential(self, x):
        """
            Evaluate potential energy at Euclidean positions x

            x, nd-array (d, m):
                Arrays of Euclidean coordinates.

            Returns:
            --------
            V, nd-array (m,):
                Values of the potential for all pairs of x-y-values.
        """
        # Transform first two dimensions to polar coordinates and compute "Lemon Slice"
        # part of the potential:
        r, phi = self._polar_rep(x[0, :], x[1, :])
        # Evaluate potential:
        V = np.cos(self.k * phi) + 1.0 / np.cos(0.5 * phi) + 10 * (r - 1) ** 2 + (1.0 / r)

        return V


    def gradient(self, x):
        """
            Evaluate gradient of potential energy at Euclidean positions x

            x, nd-array (d, m):
                Arrays of Euclidean coordinates.

            Returns:
            --------
            dx, nd-array (d, m)
                Gradient of the potential for all m data points in x.
        """
        dV = np.zeros((2, x.shape[1]))
        # Transform first two dimensions to polar coordinates and compute "Lemon Slice"
        # part of the gradient:
        r, phi = self._polar_rep(x[0, :], x[1, :])
        dV[0, :] = -(0.5 * np.sin(0.5 * phi) / np.cos(0.5 * phi) ** 2 -
                  self.k * np.sin(self.k * phi)) * (x[1, :] / r ** 2)\
                   + 20 * (r - 1) * (x[0, :] / r) - (1.0 / r ** 2) * (x[0, :] / r)
        dV[1, :] = (0.5 * np.sin(0.5 * phi) / np.cos(0.5 * phi) ** 2 -
                 self.k * np.sin(self.k * phi)) * (x[0, :] / r ** 2) \
                   + 20 * (r - 1) * (x[1, :] / r) - (1.0 / r ** 2) * (x[1, :] / r)
        return dV

    def stat_dist(self, x):
        """
            Evaluate stationary density at Euclidean positions x

            x, nd-array (d, m):
                Arrays of Euclidean coordinates.

            Returns:
            --------
            mu, nd-array (m):
                Values of the stationary density for all m data points in x.
        """
        return (1.0 / self.Z) * np.exp(-self.beta * self.potential(x))

    def _polar_rep(self, x, y):
        """
            Compute polar coordinates from 2d Euclidean coordinates:

            x, y, nd-array (m):
                Arrays of two-dimensional Euclidean coordinates to be transformed.

            Returns:
            --------
            r, phi, nd-array (m):
                Arrays of polar coordinates corresponding to x and y.
        """
        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return r, phi

