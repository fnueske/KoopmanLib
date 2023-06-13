""" This script performs cross-validation of the VAMP score for an RFF-based
 kernel approximation to analyze the dynamics of alanine dipeptide using its
 backbone dihedral angle representation.

For fixed data size of m = 20000 samples and fixed lag time, the script performs 10-fold
 cross-validation of the VAMP score for a series of kernel bandwidths and RFF feature sizes.
The resulting eigenvalues and test scores are saved to file.
"""


import numpy as np
import matplotlib.pyplot as plt

from KoopmanLib.RFF.RFF import cv_koopman_rff
from KoopmanLib.RFF.RFF_Tools import sample_rff_gauss_periodic


""" Directories: """
# Set this path to the location of the dihedral data for Alanine Dipeptide:
traj_path = "/Users/nueske/ICloud/Projekte/Test_RFF/Data/Trajectory_Data/Ala2/"
# Location to save out results:
res_dir = "/Users/nueske/ICloud/Projekte/Test_RFF/Data/Ala2/"

""" Data Settings: """
# Number of trajectories:
ntraj = 3
# Trajectory lengths:
traj_lengths = np.array([250000, 250000, 500000])
# Total number of frames:
nframes = sum(traj_lengths)
# Downsampling rate:
delta = 50
# Resulting data size:
m = int(np.sum(traj_lengths / delta))
# Number of dihedrals:
ndih = 2

""" RFF Settings: """
# Number of eigenvalues:
nev = 4
# Maximal wavenumber:
kmax = 30
# Half-period
Lper = np.pi
# tolerance for singular values of M:
cut_svd = 1e-4
# Number of tests:
ntest = 10
# Ratio of training and test data:
rtrain = .75

# Kernel bandwidths
sigma_list = np.array([1e-1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0, 5.0, 10.0])
signum = sigma_list.shape[0]
# Feature sizes:
p_list = np.array([50, 100, 200, 300, 400, 500, 1000])
pnum = p_list.shape[0]
# Lag time:
lag = 2

""" Load and downsample dihedral angle data: """
X = []
# Load one trajectory at a time:
for jj in range(ntraj):
    print("Loading trajectory %d ... "%jj)
    iX = np.load(traj_path + "Dihedrals_traj%d.npy"%jj)
    # Select the first two dihedrals (phi/psi) and downsample by delta.
    # We also ignore the first time step (= initial structure in MD run).
    X.append(iX[:ndih, 1::delta])
print("Loaded data...")
print([Xii.shape for Xii in X])


""" Run Experiments: """
# Output for eigenvalues:
d = np.zeros((signum, pnum, ntest, nev), dtype=complex)
timescales = np.zeros((signum, pnum, ntest, nev-1), dtype=complex)
dtest = np.zeros((signum, pnum, ntest), dtype=complex)


""" Score models for all values of sigma and p:"""
for ii in range(signum):
    for jj in range(pnum):
        sigma = sigma_list[ii]
        p = p_list[jj]
        print("Scoring sigma=%.2f, p=%d..."%(sigma, p))
        # Generate Fourier features:
        Omega = sample_rff_gauss_periodic(2, p, Lper, sigma, kmax)
        # Compute eigenvalues and test scores for this model:
        d_ij, dtest_ij = cv_koopman_rff(X, Omega, lag, rtrain, ntest, nev, tol=cut_svd)
        d[ii, jj, :, :] = d_ij
        timescales[ii, jj, :, :] = -(delta*lag) / np.log(np.real(d_ij[:, :-1]))
        dtest[ii, jj, :] = -dtest_ij
print("Complete.")

# Save Results:
di = {}
di["EV"] = d
di["timescales"] = timescales
di["VAMP"] = dtest
#np.savez(res_dir + "CV_Ala2_lag=%d.npz"%lag, **di)