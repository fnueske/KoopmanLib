""" This script applies the RFF-based kernel approximation to analyze stochastic
dynamics driven by the 2d Lemon Slice potential.

This script generates a single long trajectory, and estimates the leading four
eigenvalues and eigenvectors of the Koopman generator for this system, using
Random Fourier features with fixed bandwidth and fixed feature size.
"""


import numpy as np
import matplotlib.pyplot as plt

from deeptime.markov.tools.analysis.dense._pcca import _pcca_connected_isa

from KoopmanLib.Systems.LemonSlice import LemonSlice
from KoopmanLib.RFF.RFF_Tools import sample_rff_gaussian
from KoopmanLib.RFF.RFF import spectral_analysis_rff_generator


""" User settings: """
# Specify this path to the location of result files:
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

""" Simulation Settings: """
# Data size:
m = 5000
# Integration time step:
dt = 1e-3
# Saving interval:
dsave = 20
# Initial value for all simulations:
x0 = np.ones(2)

""" RFF Settings: """
# Number of random features:
p = 50
# Kernel bandwidth
sig_opt = .4
# tolerance for singular values of M:
cut_svd = 1e-4

""" Produce data and analyze using generator RFF """
# Generate new simulation:
print("Producing data...")
X = LS.Simulate(x0, (m+1) * dsave, dt)[:, dsave:-1:dsave]
# Generate fourier samples:
print("Generate RFF features...")
Omega = sample_rff_gaussian(2, p, sig_opt)
# Perform RFF spectral estimation:
print("Spectral Estimation...")
dj, Wj, M = spectral_analysis_rff_generator(X, Omega, nev, a=2.0/beta, tol=cut_svd, reversible=True)
# Uncomment this line to use non-reversible formulation:
#dj, Wj, M = spectral_analysis_rff_generator(X, Omega, nev, a=2.0/beta, b=-LS.gradient(X),
#                                            tol=cut_svd, reversible=False)

print("Eigenvalues: ...")
print(-dj[::-1])

""" Perform PCCA analysis and visualize results: """
# Evaluate eigenvectors at the data sites:
VX = np.dot(M, Wj[:, ::-1])
# Minimize imaginary part by multiplication w/ phase factor:
for ii in range(nev):
    theta_opt = 0.0
    vi = VX[:, ii]
    vmax = np.max(np.abs(np.imag(vi)))
    for theta in np.arange(0, 2*np.pi, 0.05):
        if np.max(np.abs(np.imag(np.exp(1j * theta)*vi))) < vmax:
            theta_opt = theta
            vmax = np.max(np.abs(np.imag(np.exp(1j * theta)*vi)))
    VX[:, ii] *= np.exp(1j * theta_opt)

V = VX.copy()
# Make first eigenvector constant:
V[:, 0] = np.mean(VX[:, 0])
# Apply PCCA spectral clustering:
print("Applying PCCA...")
chi, _ = _pcca_connected_isa(V, n_clusters=nev)

""" Generate figure for PCCA states:"""
print("Visualizing metastable states...")
plt.rcParams['font.size'] = 16
plt.rc('legend', fontsize=14)
plt.figure(figsize=(6, 4))
ax = plt.gca()

# Filter out data points with membership > 0.6 for each PCCA state:
cols_pcca = ["b", "r", "g", "m"]
for ii in range(nev):
    ind = np.where(chi[:, ii] >= 0.6)[0]
    sf = plt.scatter(X[0, ind], X[1, ind], c=cols_pcca[ii])

plt.title("PCCA States")
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.xlabel("x")
plt.ylabel("y")
ax.xaxis.set_label_coords(0.60, -0.03)
ax.yaxis.set_label_coords(-0.03, 0.60)
plt.show()

# Save results:
di = {}
di["X"] = X
di["VX"] = VX
di["chi"] = chi
#np.savez(res_dir + "EV_LS_m=%d_sigma_%.3f_p=%d.npz"%(m, sig_opt, p), **di)