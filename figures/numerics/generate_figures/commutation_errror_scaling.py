from __future__ import print_function
import numpy as np
import h5py
from matplotlib import rcParams
import matplotlib
# matplotlib.use('PDF')

### Text ###
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Minion Pro']
rcParams['font.size'] = 8.5
rcParams['text.usetex'] = True


### Layout ###
rcParams.update({'figure.autolayout': True})

### Axes ###
rcParams['axes.labelsize'] = 9.24994 # memoir \small for default 10pt, to match caption text
rcParams['axes.titlesize'] = 9.24994
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.labelsize'] = 9.24994
rcParams['ytick.labelsize'] = 9.24994
rcParams['lines.markersize'] = 4
rcParams['xtick.major.size'] = 2
rcParams['xtick.minor.size'] = 0
rcParams['ytick.major.size'] = 2
rcParams['ytick.minor.size'] = 0

### Legends ###
rcParams['legend.fontsize'] = 8.5 # memoir \scriptsize for default 10pt
rcParams['legend.borderpad'] = 0
rcParams['legend.handlelength'] = 1.0 # Big enough for multiple dashes or dots
rcParams['legend.handletextpad'] = 0.3
rcParams['legend.labelspacing'] = 0.3
rcParams['legend.frameon'] = False
rcParams['legend.numpoints'] = 1

# Other
rcParams['text.latex.preamble'] = ['\\usepackage{upgreek}']


colors_rgb = {'red': (200,80,80),
          'green': (80,200,130),
          'blue': (80,160,200),
          'orange': (240,170,50)}

# Convert to float colours:
colors = {name: [chan/255 for chan in value] for name, value in colors_rgb.items()}


N = 1024
b = 3
x = np.zeros((N, N), dtype=complex)

D2_6TH_ORDER_0 = -49.0/18.0
D2_6TH_ORDER_1 = 3.0/2.0
D2_6TH_ORDER_2 = -3.0/20.0
D2_6TH_ORDER_3 = 1.0/90.0


grad2_6 = np.zeros((N, N))
grad2_6 += np.diag(np.full(N-3, D2_6TH_ORDER_3), -3)
grad2_6 += np.diag(np.full(N-2, D2_6TH_ORDER_2), -2)
grad2_6 += np.diag(np.full(N-1, D2_6TH_ORDER_1), -1)
grad2_6 += np.diag(np.full(N-0, D2_6TH_ORDER_0), +0)
grad2_6 += np.diag(np.full(N-1, D2_6TH_ORDER_1), +1)
grad2_6 += np.diag(np.full(N-2, D2_6TH_ORDER_2), +2)
grad2_6 += np.diag(np.full(N-3, D2_6TH_ORDER_3), +3)


def random_banded_matrix(b):
    # Make a banded matrix:
    A = np.zeros((N, N), dtype=complex)
    for i in range(-b, b + 1):
        A += np.diag(np.random.randn(N-abs(i)), i) + 1j*np.diag(np.random.randn(N-abs(i)), i)

    return A


def get_err(args):
    A, b = args

    sizes = []
    errors = []

    for n in range(int(np.log2(N)) - 1):
        size = N/2**n
        sizes.append(size)
        B = np.zeros_like(A)
        C = np.zeros_like(A)
        i = 0
        B_regions = []
        C_regions = []
        overlap_regions = []
        while i < N:
            B_start_index = i
            B_end_index = i + size + b
            overlap1_start_index = B_end_index - b
            overlap1_end_index = B_end_index
            C_start_index = overlap1_start_index
            C_end_index = overlap1_start_index + size + b
            overlap2_start_index = C_end_index - b
            overlap2_end_index = C_end_index

            B_regions.append((B_start_index, B_end_index))
            C_regions.append((C_start_index, C_end_index))
            overlap_regions.extend([(overlap1_start_index,overlap1_end_index),
                                   (overlap2_start_index,overlap2_end_index)])

            i = C_end_index - b


        B_coeffs = np.zeros((N, N))
        C_coeffs = np.zeros((N, N))

        for start, stop in B_regions:
            B_coeffs[start:stop, start:stop] = 1

        for start, stop in C_regions:
            C_coeffs[start:stop, start:stop] = 1

        for start, stop in overlap_regions:
            B_coeffs[start:stop, start:stop] /= 2
            C_coeffs[start:stop, start:stop] /= 2

        B = B_coeffs * A
        C = C_coeffs * A

        assert np.allclose(A, B+C)

        err = np.sqrt((np.abs(np.dot(B, C) - np.dot(C, B))**2).mean())

        print(2**n, err)
        errors.append(err)

    sizes = np.array(sizes, dtype=float)
    errors = np.array(errors, dtype=float)

    return sizes, errors

import matplotlib.pyplot as plt

if __name__ == '__main__':
    from multiprocessing import Pool

    grad2_6_size, grad2_6_err = get_err((grad2_6, 3))

    pool = Pool(10)
    # sizes, errors = get_err(random_banded_matrix(b))
    with h5py.File('commutation_error_scaling.h5') as f:
        for b in [1, 2, 3]:
            results = pool.map(get_err, [(random_banded_matrix(b), b) for i in range(20)])
            mean_err = np.mean([errors for sizes, errors in results], axis=0)
            std_err = np.std([errors for sizes, errors in results], axis=0)
            sizes = results[0][0]
            f[f'{b}:mean_err'] = mean_err
            f[f'{b}:std_err'] = std_err
        f[f'sizes'] = sizes
    
    # plt.loglog(sizes, mean_err, 'b-')
    plt.loglog(grad2_6_size, grad2_6_err, 'bo-', label=R'$\nabla^2$ (FD)')
    plt.fill_between(sizes, (mean_err - std_err), (mean_err + std_err),
                     alpha=0.5, label='random ($1\sigma$ range)')
    plt.plot(sizes, mean_err, 'bo')

    plt.plot(sizes, 1/np.sqrt(sizes), 'k--', label=R'$s^{-\frac{1}{2}}$')
    # plt.plot(sizes, mean_err[-1] * np.sqrt(sizes[-1]/sizes), 'k--')
    # plt.errorbar(nums, errs/errs[-1], yerr=errs/errs[-1]/np.sqrt(nums))
    
    plt.xlabel('submatrix size $s$')
    plt.ylabel(R'Commutation error $\left(N^{-2}\sum_{ij} |\left[B, C\right]_{ij}|^2\right)^{-\frac{1}{2}}$')
    plt.legend()
    plt.show()

