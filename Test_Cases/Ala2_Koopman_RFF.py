""" This script applies the RFF-based kernel approximation to analyze the
dynamics of the alanine dipeptide using its backbone dihedral angle representation.

We estimate the leading four eigenvalues and eigenvectors of the Koopman operator for
this system, using Random Fourier features with fixed bandwidth and fixed feature size.
"""

import numpy as np
import matplotlib.pyplot as plt

from KoopmanLib.RFF.RFF_Tools import sample_rff_gauss_periodic
from KoopmanLib.RFF.RFF import spectral_analysis_rff_koopman

from deeptime.markov.tools.analysis.dense._pcca import _pcca_connected_isa


""" Directories: """
# Set this path to the location of the dihedral data for Alanine Dipeptide:
traj_path = "/Users/nueske/ICloud/Projekte/Test_RFF/Data/Trajectory_Data/Ala2/"
# Set this path to the location of result files:
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
# Number of random features:
p = 50
# Kernel bandwidth
sig_opt = .6
# Select the lag time:
lag = 2
# Maximal wavenumber:
kmax = 30
# Half-period
Lper = np.pi
# tolerance for singular values of M:
cut_svd = 1e-4

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

""" Spectral analysis based on RFF """
# Generate Fourier features:
print("Generate Fourier features...")
Omega = sample_rff_gauss_periodic(2, p, Lper, sig_opt, kmax)
# Compute eigenvalues and eigenvectors:
print("Spectral Analysis...")
di, Wi, M = spectral_analysis_rff_koopman(X, Omega, lag, nev, tol=cut_svd)
print("Eigenvalues...")
print(di)
print("Timescales...")
print(-delta*lag / np.log(np.real(di[:-1])))

""" PCCA Analysis of Eigenvctors: """
print("PCCA analysis...")
# Evaluate eigenvectors at all data sites:
W = M @ Wi

# Minimize imaginary part by multiplication w/ phase factor:
for ii in range(nev):
    theta_opt = 0.0
    vi = W[:, ii]
    vmax = np.max(np.abs(np.imag(vi)))
    for theta in np.arange(0, 2*np.pi, 0.05):
        if np.max(np.abs(np.imag(np.exp(1j * theta)*vi))) < vmax:
            theta_opt = theta
            vmax = np.max(np.abs(np.imag(np.exp(1j * theta)*vi)))
    W[:, ii] *= np.exp(1j * theta_opt)

# Make first eigenvector constant:
V = np.real(W[:, ::-1]).copy()
V[:, 0] = np.mean(V[:, 0])
# Compute PCCA decomposition:
chi, _ = _pcca_connected_isa(V[:, :nev], n_clusters=nev)

""" Plot PCCA states: """
plt.rcParams['font.size'] = 16
plt.rc('legend', fontsize=10)
plt.figure(figsize=(6, 4))
ax = plt.gca()

# Stack all time-lagged dihedral trajectories for PCCA visualization:
Xall = np.hstack([X[rr][:, :-lag] for rr in range(ntraj)])

cols_pcca = ["b", "r", "g", "m"]
# For each PCCA state, extract all frames with membership >=0.6,
# and scatter these data points:
for ii in range(nev):
    ind = np.where(chi[:, ii] >= 0.6)[0]
    sf = plt.scatter(Xall[0, ind], Xall[1, ind], c=cols_pcca[ii])

plt.title("PCCA States")
plt.xlim([-np.pi, np.pi])
plt.ylim([-np.pi, np.pi])
plt.xlabel(r"$\phi$")
plt.ylabel(r"$\psi$")
ax.xaxis.set_label_coords(0.60, -0.03)
ax.yaxis.set_label_coords(-0.03, 0.60)

# Save results:
print("Saving results...")
di = {}
di["Xall"] = Xall
di["V"] = V
di["chi"] = chi
#np.savez(res_dir + "EV_Ala2_m=%d_sigma_%.3f_p=%d.npz"%(m, sig_opt, p), **di)

plt.show()