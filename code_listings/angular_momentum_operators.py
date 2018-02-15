import numpy as np
hbar = 1.054571628e-34

def angular_momentum_operators(J):
    """Construct matrix representations of the angular momentum operators Jx,
    Jy, Jz and J2 in the eigenbasis of Jz for given total angular momentum
    quantum number J. Return them, as well as the number of angular momentum
    projection states, a list of angular momentum projection quantum numbers
    mJ, and a list of their corresponding eigenvectors, in the same order as
    the matrix elements (in descending order of mJ)."""
    n_mJ = int(round(2*J + 1))
    mJlist = np.linspace(J, -J, n_mJ)
    Jp = np.diag([hbar * np.sqrt(J*(J+1) - mJ*(mJ + 1)) for mJ in mJlist if mJ < J], 1)
    Jm = np.diag([hbar * np.sqrt(J*(J+1) - mJ*(mJ - 1)) for mJ in mJlist if mJ > -J], -1)
    Jx = (Jp + Jm) / 2
    Jy = (Jp - Jm) / 2j
    Jz = np.diag([hbar*mJ for mJ in mJlist])
    J2 = Jx**2 + Jy**2 + Jz**2
    basisvecs_mJ = [vec for vec in np.identity(n_mJ)]
    return Jx, Jy, Jz, J2, n_mJ, mJlist, basisvecs_mJ
