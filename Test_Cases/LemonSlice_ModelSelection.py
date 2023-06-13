""" This script performs cross-validation of the VAMP score for an RFF-based
 kernel approximation to analyze stochastic dynamics driven by the 2d Lemon Slice potential.

For fixed data size, the script perform 20-fold cross-validation of the VAMP score for a
series of kernel bandwidths and RFF feature sizes. The resulting eigenvalues and test scores
are saved to file.
"""


import numpy as np
import matplotlib.pyplot as plt

from KoopmanLib.RFF.RFF import cv_generator_rff
from KoopmanLib.RFF.RFF_Tools import sample_rff_gaussian
from KoopmanLib.Systems.LemonSlice import LemonSlice


""" User settings: """
# Location to save out results:
res_dir = "/Users/nueske/ICloud/Projekte/Test_RFF/Data/LS/"

""" System Settings: """
# Pre-factors
Cp = 4.0
# Inverse temperature:
beta = 1.0
# Number of rates to calculate:
nev = 4
# Instantiate system:
LS = LemonSlice(Cp, beta=beta)

""" Settings regarding data: """
# Initial value for all simulations:
x0 = np.ones(2)
# Integration time step:
dt = 1e-3
# Saving interval:
dsave = 20

""" RFF Settings: """
# Data size:
m = 5000
# Kernel bandwidths
sigma_list = np.array([1e-2, 5e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0])
signum = sigma_list.shape[0]
# Feature sizes:
p_list = np.array([50, 100, 200, 300, 400, 500])
pnum = p_list.shape[0]
# tolerance for singular values of M:
cut_svd = 1e-4
# Number of eigenvalues:
nev = 4
# Number of tests:
ntest = 20
# Ration of training and test data:
rtrain = .75


""" Run Experiments: """
# Output for eigenvalues:
d = np.zeros((signum, pnum, ntest, nev), dtype=complex)
dtest = np.zeros((signum, pnum, ntest), dtype=complex)

# Generate data:
print("Generate data...")
X = LS.Simulate(x0, (m + 1) * dsave, dt)[:, dsave:-1:dsave]

""" Score models for all values of sigma and p:"""
for ii in range(signum):
    for jj in range(pnum):
        sigma = sigma_list[ii]
        p = p_list[jj]
        print("Scoring sigma=%.2f, p=%d..."%(sigma, p))
        # Generate Fourier features:
        Omega = sample_rff_gaussian(2, p, sigma)
        # Compute eigenvalues and test scores for this model:
        d_ij, dtest_ij = cv_generator_rff(X, Omega, (2.0/beta), rtrain, ntest, nev, tol=cut_svd)
        d[ii, jj, :, :] = d_ij
        dtest[ii, jj, :] = -dtest_ij
print("Complete.")

# Save Results:
di = {}
di["EV"] = d
di["VAMP"] = dtest
#np.savez(res_dir + "CV_LS_m=%d.npz" % m, **di)