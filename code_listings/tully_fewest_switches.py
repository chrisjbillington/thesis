import numpy as np

def S_matrix_fewest_switches(psi, U):
    """Compute the fewest-switches S matrix for an array psi, shape (M, N),
    containing N state vectors for an system with M states, and the array of
    corresponding unitaries unitaries U, shape (M, M, N), describing the evolution
    of the state vectors over a short time interval."""
    P = 2 * np.einsum('in,ijn,jn->ijn', psi.conj(), U, psi).real
    P[P < 0] = 0
    S = P / np.abs(psi**2)
    S_diags = np.einsum('iin->in', S)
    S_diags[...] = 0
    S_diags[...] = 1 - np.sum(S, axis=0)
    return S