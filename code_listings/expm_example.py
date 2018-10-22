import numpy as np
from numpy.linalg import eigh

def expiH(H):
    """compute exp(i*H), where H, shape (..., N, N) is an array of N by N Hermitian
    matrices, using the diagonalisation method, and where i is the imaginary unit. This
    fucntion is useful because scipy's expm can't take an array of matrices as input, it
    can only do one at a time."""

    # Diagonalise the matrices:
    evals, evecs = eigh(H)

    # Now we compute exp(i*H) = Q exp(i*D) Q^\dagger where Q is the matrix of
    # eigenvectors (as columns) and D is the diagonal matrix of eigenvalues:

    Q = evecs
    Q_dagger = Q.conj().swapaxes(-1, -2) # Only transpose the matrix dimensions
    exp_iD_diags = np.exp(1j*evals)

    # Compute the 3-term matrix product Q*exp_iD_diags*Q_dagger using the
    # einsum function in order to specify which array axes of each array to
    # sum over:
    return np.einsum('...ik,...k,...kj->...ij', Q, exp_iD_diags, Q_dagger)