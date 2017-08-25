# To be run with mpi like so:
#     mpirun -n <N_PROCESSES> python vortex_test.py

from __future__ import division, print_function
import sys
import numpy as np

# Simulation size set by command line argument
OUTPUT_INTERVAL = 100

from BEC2D import Simulator2D

# Constants:
pi = np.pi
hbar = 1.054571726e-34
a_0  = 5.29177209e-11                       # Bohr radius
u    = 1.660539e-27                         # unified atomic mass unit
m  = 86.909180*u                            # 87Rb atomic mass
a  = 98.98*a_0                              # 87Rb |2,2> scattering length
g  = 4*pi*hbar**2*a/m                       # 87Rb self interaction constant
rho_max = 2.5e14*1e6                        # Desired maximum density
R = 7.5e-6      # Desired Thomas-Fermi radius
omega = np.sqrt(2*g*rho_max/(m*R**2))       # Trap frequency  corresponding to desired density and radius
mu = g*rho_max                              # Desired chemical potential of the groundstate
N_2D = pi*rho_max*R**2/2                    # Thomas Fermi estimate of atom number for given chemical potential.
healing_length = 1/np.sqrt(8*pi*a*rho_max)

# Total spatial region over all MPI processes:
x_min_global = -10e-6
x_max_global = 10e-6
y_min_global = -10e-6
y_max_global = 10e-6

# Number of DVR basis functions per element:
Nx = 17
Ny = 17

# # We have to be able to resolve the healing length. What spatial resolution do
# # we require to have ~4 points per healing length?
# nx_total = 10*(x_max_global - x_min_global)/healing_length

# Finite elements:
n_elements_x_global = 16
n_elements_y_global = 16

# Number of components to the wavefunction:
n_components = 1

simulator = Simulator2D(x_min_global, x_max_global, y_min_global, y_max_global,
                        n_elements_x_global, n_elements_y_global, Nx, Ny, n_components,
                        periodic_x=False, periodic_y=False,
                        output_filepath='vortex_test_softclipped.h5')
x = simulator.x
y = simulator.y

# Kinetic energy operators:
Kx = -hbar**2/(2*m) * simulator.grad2x
Ky = -hbar**2/(2*m) * simulator.grad2y

# V = 0.5*m*omega**2*(x**2 + y**2)
r2 = x**2.0 + y**2.0
r  = np.sqrt(r2)
alpha = 2
V = 0.5 * m * omega**2 * R**2.0 * (r/R)**alpha
# V[:] = 0
# The trap rotation frequency:
Omega = 0.9 * omega * 0

dx_min = np.diff(x[0, 0, :, 0, 0, 0]).min()
dy_min = np.diff(y[0, 0, :, 0, 0]).min()

dx_max = np.diff(x[0, 0, :, 0, 0, 0]).max()
print(dx_max)

dispersion_timescale = min(dx_min, dy_min)**2 * m / (pi * hbar)
chemical_potential_timescale = 2*pi*hbar/mu
potential_timescale = 2*pi*hbar/V.max()


def H(t, psi, *slices):
    x_elements, y_elements, x_points, y_points = slices
    Kx = -hbar**2/(2*m) * simulator.grad2x[x_points]
    Rx = - Omega * 1j*hbar * y[y_elements, :, y_points] * simulator.gradx[x_points]
    Ky = -hbar**2/(2*m) * simulator.grad2y[y_points]
    Ry = + Omega*1j*hbar*x[x_elements, :, x_points] * simulator.grady[y_points]
    U = V[slices]
    U_nonlinear = g * psi[slices].conj() * simulator.density_operator[x_points, y_points] * psi[slices]
    return Kx + Rx, Ky + Ry, U, U_nonlinear


def initial_guess(x, y):
    sigma_x = 0.5*R
    sigma_y = 0.5*R
    f = np.sqrt(np.exp(-x**2/(2*sigma_x**2) - y**2/(2*sigma_y**2)))
    # f = np.random.random(f.shape) + 1j*np.random.random(f.shape)
    return f


def soft_clip(a, soft_max, hard_max):
    a_out = np.zeros(a.shape)
    clip = (a >= soft_max)
    a_out[~clip] = a[~clip]

    log_a = np.log(a[clip])
    c = np.log(hard_max)
    A = np.log(soft_max) - np.log(hard_max)
    log_a0 = np.log(soft_max)
    k = 1/(np.log(hard_max) - np.log(soft_max))

    log_a_out_clip = c + A*np.exp(-k*(log_a - log_a0))
    a_out[clip] = np.exp(log_a_out_clip)

    return a_out


def run_sims():

    psi = simulator.elements.make_vector(initial_guess)
    simulator.normalise(psi, N_2D)

    psi = simulator.find_groundstate(psi, H, mu, relaxation_parameter=1.7, output_group='initial', convergence=1e-14)

    # # Scatter some vortices randomly about.
    # # Ensure all MPI tasks agree on the location of the vortices, by
    # # seeding the pseudorandom number generator with the same seed in
    # # each process:
    # np.random.seed(42)
    # for i in range(30):
    #     sign = np.sign(np.random.normal())
    #     x_vortex = np.random.normal(0, scale=R)
    #     y_vortex = np.random.normal(0, scale=R)
    #     psi[:] *= np.exp(sign * 1j*np.arctan2(simulator.y - y_vortex, simulator.x - x_vortex))

    # dt = 2e-6

    # # A little imaginary time evolution to smooth the newly created vortices:
    # psi = simulator.evolve(psi, H, dt=dt/2, t_final=chemical_potential_timescale,
    #                        imaginary_time=True, mu=mu, method='rk4', output_group='smoothing')

    # # Actual time evolution:
    # psi = simulator.evolve(psi, H, dt=dt, t_final=10e-3, mu = mu,
    #                        output_group='evolution',
    #                        method='rk4', output_interval=1, wavefunction_output=False)


if __name__ == '__main__':
    run_sims()
