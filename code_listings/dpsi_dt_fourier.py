import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

pi = np.pi
u  = 1.660539e-27 # unified atomic mass unit
m  = 86.909180*u # 87Rb atomic mass
omega = 15 # Harmonic trap frequency
hbar = 1.054571726e-34 # Reduced Planck's constant

# Space:
nx = ny = 256
x_max = y_max = 100e-6

# Arrays of components of position vectors. The reshaping is to ensure that
# when used in arithmetic with each other, these arrays will be treated as if
# they are two dimensional with repeated values along the dimensions of size
# 1, up to the size of the other array (this is called broadcasting in numpy).
x = np.linspace(-x_max, x_max, nx, endpoint=False).reshape(1, nx)
y = np.linspace(-y_max, y_max, ny, endpoint=False).reshape(ny, 1)

# Grid spacing:
dx = x[0, 1] - x[0, 0]

# Arrays of components of k vectors.
kx = 2*pi*fftfreq(nx, d=dx).reshape(1, nx)
ky = 2*pi*fftfreq(nx, d=dx).reshape(ny, 1)

# The kinetic energy operator in Fourier space (shape ny, nx).
K_fourier = hbar**2 * (kx**2 + ky**2)/(2*m)

# The potential operator in real space (shape ny, nx)
V_real = 0.5 * m * omega**2 * (x**2 + y**2)

def dpsi_dt(t, psi):
    """Return a 2D array for  the time derivative of the 2D array psi
    representing a discretised wavefucntion obeying the Schrodinger wave
    equation"""
    K_real_psi = ifft2(K_fourier * fft2(psi))
    return -1j/hbar * (K_real_psi + V_real * psi)