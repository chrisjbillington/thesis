from __future__ import print_function
import numpy as np
import h5py
from matplotlib import rcParams
import matplotlib
matplotlib.use('PDF')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

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
rcParams['legend.handlelength'] = 1.5 # Big enough for multiple dashes or dots
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

random_color = colors['red']
grad_color = colors['orange']
grad2_color = colors['blue']

random_marker = 'v'
grad_marker = 's'
grad2_marker = 'o'



N = 1024
b = 3
x = np.zeros((N, N), dtype=complex)

# Constants for central finite differences:
D_2ND_ORDER_1 = 1.0/2.0

D_4TH_ORDER_1 = 2.0/3.0
D_4TH_ORDER_2 = -1.0/12.0

D_6TH_ORDER_1 = 3.0/4.0
D_6TH_ORDER_2 = -3.0/20.0
D_6TH_ORDER_3 = 1.0/60.0

D2_2ND_ORDER_0 = -2.0
D2_2ND_ORDER_1 = 1.0

D2_4TH_ORDER_0 = -5.0/2.0
D2_4TH_ORDER_1 = 4.0/3.0
D2_4TH_ORDER_2 = -1.0/12.0

D2_6TH_ORDER_0 = -49.0/18.0
D2_6TH_ORDER_1 = 3.0/2.0
D2_6TH_ORDER_2 = -3.0/20.0
D2_6TH_ORDER_3 = 1.0/90.0


# Finite differences matrices - first and second derivatives to second, fourth
# and sixth order:

grad_2 = np.zeros((N, N))
grad_2 += np.diag(np.full(N-1, D_2ND_ORDER_1), -1)
grad_2 += np.diag(np.full(N-1, D_2ND_ORDER_1), +1)

grad2_2 = np.zeros((N, N))
grad2_2 += np.diag(np.full(N-1, D2_2ND_ORDER_1), -1)
grad2_2 += np.diag(np.full(N-0, D2_2ND_ORDER_0), +0)
grad2_2 += np.diag(np.full(N-1, D2_2ND_ORDER_1), +1)

grad_4 = np.zeros((N, N))
grad_4 += np.diag(np.full(N-2, D_4TH_ORDER_2), -2)
grad_4 += np.diag(np.full(N-1, D_4TH_ORDER_1), -1)
grad_4 += np.diag(np.full(N-1, D_4TH_ORDER_1), +1)
grad_4 += np.diag(np.full(N-2, D_4TH_ORDER_2), +2)

grad2_4 = np.zeros((N, N))
grad2_4 += np.diag(np.full(N-2, D2_4TH_ORDER_2), -2)
grad2_4 += np.diag(np.full(N-1, D2_4TH_ORDER_1), -1)
grad2_4 += np.diag(np.full(N-0, D2_4TH_ORDER_0), +0)
grad2_4 += np.diag(np.full(N-1, D2_4TH_ORDER_1), +1)
grad2_4 += np.diag(np.full(N-2, D2_4TH_ORDER_2), +2)

grad_6 = np.zeros((N, N))
grad_6 += np.diag(np.full(N-3, D_6TH_ORDER_3), -3)
grad_6 += np.diag(np.full(N-2, D_6TH_ORDER_2), -2)
grad_6 += np.diag(np.full(N-1, D_6TH_ORDER_1), -1)
grad_6 += np.diag(np.full(N-1, D_6TH_ORDER_1), +1)
grad_6 += np.diag(np.full(N-2, D_6TH_ORDER_2), +2)
grad_6 += np.diag(np.full(N-3, D_6TH_ORDER_3), +3)

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


def generate_data():
    from multiprocessing import Pool

    _, grad_2_err = get_err((grad_2, 1))
    _, grad2_2_err = get_err((grad2_2, 1))

    _, grad_4_err = get_err((grad_4, 2))
    _, grad2_4_err = get_err((grad2_4, 2))

    _, grad_6_err = get_err((grad_6, 3))
    _, grad2_6_err = get_err((grad2_6, 3))

    pool = Pool(10)

    with h5py.File('commutation_error_scaling.h5') as f:
        for b in [1, 2, 3]:
            results = pool.map(get_err, [(random_banded_matrix(b), b) for i in range(20)])
            mean_err = np.mean([errors for sizes, errors in results], axis=0)
            std_err = np.std([errors for sizes, errors in results], axis=0)
            sizes = results[0][0]
            f[f'{b}:mean_err'] = mean_err
            f[f'{b}:std_err'] = std_err

        f[f'sizes'] = sizes

        f['grad_2_err'] = grad_2_err
        f['grad2_2_err'] = grad2_2_err
        f['grad_4_err'] = grad_4_err
        f['grad2_4_err'] = grad2_4_err
        f['grad_6_err'] = grad_6_err
        f['grad2_6_err'] = grad2_6_err


def make_figure():
    with h5py.File('commutation_error_scaling.h5', 'r') as f:
        mean_err_random_b1 = f['1:mean_err'][:]
        std_err_random_b1 = f['1:std_err'][:]
        mean_err_random_b2 = f['2:mean_err'][:]
        std_err_random_b2 = f['2:std_err'][:]
        mean_err_random_b3 = f['3:mean_err'][:]
        std_err_random_b3 = f['3:std_err'][:]
        grad_2_err = f['grad_2_err'][:]
        grad2_2_err = f['grad2_2_err'][:]
        grad_4_err = f['grad_4_err'][:]
        grad2_4_err = f['grad2_4_err'][:]
        grad_6_err = f['grad_6_err'][:]
        grad2_6_err = f['grad2_6_err'][:]
        sizes = f['sizes'][:]


    FIG_WIDTH = 4.5
    FIG_HEIGHT = 2.25

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = gridspec.GridSpec(1, 3, left=0.12, bottom=0.12,
                           right=0.985, top=0.95, wspace=0.075, hspace=0.075)

    plt.subplot(gs[:,0])
    plt.gca().tick_params(direction='in')
    plt.loglog(sizes[1:], 1/np.sqrt(sizes[1:]), 'k--', label=R'$s^{-\frac{1}{2}}$')
    plt.fill_between(sizes,
                     (mean_err_random_b1 - std_err_random_b1),
                     (mean_err_random_b1 + std_err_random_b1),
                     facecolor=random_color,
                     alpha=0.5)

    plt.plot(sizes, mean_err_random_b1, marker=random_marker, linestyle='',
             color=random_color, markeredgecolor=random_color, label='random')

    plt.loglog(sizes, grad2_2_err, marker=grad2_marker,
             color=grad2_color, markeredgecolor=grad2_color, label=R'$\partial_x^2$ 2$^{\rm nd}$ order \textsc {fd}')
    plt.loglog(sizes, grad_2_err, marker=grad_marker,
             color=grad_color, markeredgecolor=grad_color, label=R'$\partial_x$ 2$^{\rm nd}$ order \textsc {fd}')

    plt.axis([1, 2000, 0.00003, 1])
    plt.annotate(R'$b=1$', xy=(0.75, 0.925), xycoords='axes fraction')

    plt.ylabel(R'\textsc{rms} commutator matrix element')

    # plt.ylabel(R'commutation error $\left(N^{-2}\sum_{ij} '
    #            R'|\left[B, C\right]_{ij}|^2\right)^{-\frac{1}{2}}$')

    plt.legend()

    plt.subplot(gs[:,1])
    plt.gca().tick_params(direction='in')
    plt.loglog(sizes[1:], 1/np.sqrt(sizes[1:]), 'k--', label=R'$s^{-\frac{1}{2}}$')
    plt.fill_between(sizes,
                     (mean_err_random_b2 - std_err_random_b2),
                     (mean_err_random_b2 + std_err_random_b2),
                     facecolor=random_color,
                     alpha=0.5)
    plt.plot(sizes, mean_err_random_b2, marker=random_marker, linestyle='',
             color=random_color, markeredgecolor=random_color, label='random')

    plt.loglog(sizes, grad2_4_err, marker=grad2_marker,
             color=grad2_color, markeredgecolor=grad2_color, label=R'$\partial_x^2$ 4$^{\rm th}$ order \textsc {fd}')
    plt.loglog(sizes, grad_4_err, marker=grad_marker,
             color=grad_color, markeredgecolor=grad_color, label=R'$\partial_x$ 4$^{\rm th}$ order \textsc {fd}')

    plt.axis([1, 2000, 0.00003, 1])
    plt.legend()
    plt.gca().yaxis.set_ticklabels([])
    plt.annotate(R'$b=2$', xy=(0.75, 0.925), xycoords='axes fraction')

    plt.subplot(gs[:,2])
    plt.gca().tick_params(direction='in')
    plt.loglog(sizes[1:], 1/np.sqrt(sizes[1:]), 'k--', label=R'$s^{-\frac{1}{2}}$')
    plt.fill_between(sizes,
                     (mean_err_random_b3 - std_err_random_b3),
                     (mean_err_random_b3 + std_err_random_b3),
                     facecolor=random_color,
                     alpha=0.5)
    plt.plot(sizes, mean_err_random_b3, marker=random_marker, linestyle='',
             color=random_color, markeredgecolor=random_color, label='random')

    plt.loglog(sizes, grad2_6_err, marker=grad2_marker,
             color=grad2_color, markeredgecolor=grad2_color,  label=R'$\partial_x^2$ 6$^{\rm th}$ order \textsc {fd}')
    plt.loglog(sizes, grad_6_err, marker=grad_marker,
             color=grad_color, markeredgecolor=grad_color, label=R'$\partial_x$ 6$^{\rm th}$ order \textsc {fd}')

    plt.axis([1, 2000, 0.00003, 1])
    plt.legend()
    plt.gca().yaxis.set_ticklabels([])
    plt.annotate(R'$b=3$', xy=(0.75, 0.925), xycoords='axes fraction')

    fig.text(0.455, 0.03, 'submatrix size $s$', va='center')
    

    
    # plt.show()
    plt.savefig('../commutation_error.pdf')

if __name__ == '__main__':
    # generate_data()
    make_figure()