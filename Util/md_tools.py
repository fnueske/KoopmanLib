import numpy as np
import scipy.linalg as scl

""" This module contains specific functions to process molecular dynamics 
    simulation data. Currently available functionality includes:
    
    compute_dihedral_jacobian:
            Compute the Jacobian of a set of dihedral angles w.r.t. all
            atomic coordinates.
    dih_grad:
            Evaluate the gradient of a single dihedral w.r.t. the twelve atomic
            coordinates defining the dihedral
"""

def compute_dihedral_jacobian(xyz, atom_ind, output_freq=None):
    """ Compute the Jacobian of a set of dihedral angles w.r.t. all
        atomic coordinates involved in the definition of these dihedrals.

    Parameters:
    -----------
    xyz, ndarray(m, natoms, 3):     m snapshots of the Euclidean coordinates of natoms
                                    atoms in a molecule.
    atom_ind, ndarray(ndih, 4):     list of the indices of the four atoms spanning each
                                    dihedral in xyz.
    output_freq, int (optional):   Print out a message every output_freq'th step.
    
    Returns:
    -------
    grad_dih (ndih, natoms, 3, m):  gradients of all ndih dihedrals with respect to all
                                    atomic coordinates, for all m time steps.
     """
    # Get info:
    m, natoms, _ = xyz.shape
    ndih = atom_ind.shape[0]
    # Prepare output:
    grad_dih = np.zeros((ndih, natoms, 3, m))
    # Compute gradients for all dihedrals and all time steps:
    for ll in range(m):
        for jj in range(ndih):
            grad_dih[jj, atom_ind[jj, :], :, ll] = dih_grad(xyz[ll, atom_ind[jj, :], :])
        if (output_freq is not None) and (np.remainder(ll + 1, output_freq) == 0):
            print("Completed %d snapshots." % (ll + 1))

    return grad_dih

""" Evaluate the gradient of a single dihedral w.r.t. the twelve atomic
    coordinates defining the dihedral.
"""
# TODO: add reference for formula for the dihedral gradient
def dih_grad(r):
    """
    Parameters:
    ------------
    r, ndarray(4, 3): position vectors of four atoms.

    Returns:
    ------------
    grad, ndarray(4, 3): gradient of the dihedral w.r.t. all Euclidean coords.

    """
    # Extract position vectors of individual atoms:
    ri = r[0, :]
    rj = r[1, :]
    rk = r[2, :]
    rl = r[3, :]
    # Assemble gradient one coordinate at a time:
    grad = np.zeros((4, 3))
    grad[0, :] = _m1(ri, rj, rk, rl)
    grad[1, :] = ((_S1(ri, rj, rk, rl) - 1) * _m1(ri, rj, rk, rl)
                  + _S2(ri, rj, rk, rl) * _m2(ri, rj, rk, rl))
    grad[2, :] = -(_S1(ri, rj, rk, rl) * _m1(ri, rj, rk, rl) +
                   (_S2(ri, rj, rk, rl) - 1) * _m2(ri, rj, rk, rl))
    grad[3, :] = -_m2(ri, rj, rk, rl)
    return grad

""" Helper Functions to calculate Jacobian of dihedral angles: """
# Distances and distance vectors:
def _dist_vec(ri, rj):
    return ri - rj

def _dist(ri, rj):
    return scl.norm(ri - rj)

# Scalar products:
def _S1(ri, rj, rk, rl):
    rkj = _dist(rk, rj)
    return np.dot(_dist_vec(ri, rj), _dist_vec(rk, rj)) / (rkj ** 2)

def _S2(ri, rj, rk, rl):
    rkj = _dist(rk, rj)
    return np.dot(_dist_vec(rk, rl), _dist_vec(rk, rj)) / (rkj ** 2)

# Normal vectors:
def _m_vec(ri, rj, rk, rl):
    return np.cross(_dist_vec(ri, rj), _dist_vec(rk, rj))

def _n_vec(ri, rj, rk, rl):
    return np.cross(_dist_vec(rk, rj), _dist_vec(rk, rl))

# Unit normal vectors:
def _m_vec_norm(ri, rj, rk, rl):
    m = _m_vec(ri, rj, rk, rl)
    return m / (scl.norm(m)**2)

def _n_vec_norm(ri, rj, rk, rl):
    n = _n_vec(ri, rj, rk, rl)
    return n / (scl.norm(n)**2)

def _m1(ri, rj, rk, rl):
    rkj = _dist(rk, rj)
    mn = _m_vec_norm(ri, rj, rk, rl)
    return rkj*mn

def _m2(ri, rj, rk, rl):
    rkj = _dist(rk, rj)
    nn = _n_vec_norm(ri, rj, rk, rl)
    return rkj*nn