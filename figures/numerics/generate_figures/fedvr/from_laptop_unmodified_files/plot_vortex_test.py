# To be run as a single process, not with MPI:
#     python plot_vortex_text.py

from __future__ import division, print_function
import os
import numpy as np
import h5py
from BEC2D import Simulator2D
import matplotlib
matplotlib.use('Agg')
import matplotlib.image
pi = np.pi



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



def plot(i, psi):
    x_plot, y_plot, psi_interp = simulator.elements.interpolate_vector(psi, Nx, Ny)
    rho = (psi.conj() * simulator.density_operator * psi).real
    rho = rho.transpose(0, 2, 1, 3, 4, 5).reshape((32*7, 32*7))
    # psi_interp = psi.transpose(0, 2, 1, 3, 4, 5).reshape((32*7, 32*7))
    # psi_interp[x_plot**2 + y_plot**2 < 11e-6**2] = 0
    # psi_interp = psi_interp.transpose()
    # psi_interp = psi.transpose(0, 2, 1, 3, 4, 5).reshape((32*7, 32*7))
    # rho = np.abs(psi_interp)**2

    phase = np.angle(psi_interp)

    hsl = np.zeros(psi_interp.shape + (3,))
    hsl[:, :, 2] = rho/rho.max()
    rgb = matplotlib.colors.hsv_to_rgb(hsl)
    hsl[:, :, 0] = np.array((phase + pi)/(2*pi))
    hsl[:, :, 1] = 0.33333
    rgb = matplotlib.colors.hsv_to_rgb(hsl)
    matplotlib.image.imsave('evolution/%04d.png' % i,  np.log(rho.transpose()), origin='lower')


if __name__ == '__main__':
    if not os.path.isdir('evolution'):
        os.mkdir('evolution')
    with h5py.File('vortex_test.h5/0.h5', 'r') as f:
        x_min_global = f.attrs['x_min_global']
        x_max_global = f.attrs['x_max_global']
        y_min_global = f.attrs['y_min_global']
        y_max_global = f.attrs['y_max_global']
        n_elements_x_global, n_elements_y_global, Nx, Ny, n_components, _ = f.attrs['global_shape']

        simulator = Simulator2D(x_min_global, x_max_global, y_min_global, y_max_global,
                                n_elements_x_global, n_elements_y_global, Nx, Ny, n_components)

        x = simulator.x
        y = simulator.y


        r2 = x**2.0 + y**2.0
        r  = np.sqrt(r2)
        alpha = 30
        V = 0.5 * m * omega**2 * R**2.0 * (r/R)**alpha

        dx_min = np.diff(x[0, 0, :, 0, 0, 0]).min()
        dy_min = np.diff(y[0, 0, :, 0, 0]).min()
        dt = min(dx_min, dy_min)**2 * m / (8 * hbar)

        Vmax = 10*hbar/dt
        out_of_bounds = V > Vmax
        V[out_of_bounds] = Vmax

        i = 5000
        while True:
            psi = f['/output/evolution/psi'][i]
            print(i)
            psi[out_of_bounds] = 0
            plot(i, psi)
            i += 1
