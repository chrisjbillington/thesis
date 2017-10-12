from __future__ import division
from collections import defaultdict
import numpy as np

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

def make_fd_operator(Npts, order):
    # Constants for central finite differences:
    D2_2ND_ORDER_0 = -2.0
    D2_2ND_ORDER_1 = 1.0

    D2_4TH_ORDER_0 = -5.0/2.0
    D2_4TH_ORDER_1 = 4.0/3.0
    D2_4TH_ORDER_2 = -1.0/12.0

    D2_6TH_ORDER_0 = -49.0/18.0
    D2_6TH_ORDER_1 = 3.0/2.0
    D2_6TH_ORDER_2 = -3.0/20.0
    D2_6TH_ORDER_3 = 1.0/90.0

    D2_8TH_ORDER_0 = -14350.0/5040
    D2_8TH_ORDER_1 = 8064.0/5040
    D2_8TH_ORDER_2 = -1008.0/5040
    D2_8TH_ORDER_3 = 128.0/5040
    D2_8TH_ORDER_4 = -9.0/5040

    D2_10TH_ORDER_0 = -73766/25200
    D2_10TH_ORDER_1 = 42000/25200
    D2_10TH_ORDER_2 = -6000/25200
    D2_10TH_ORDER_3 = 1000/25200
    D2_10TH_ORDER_4 = -125/25200
    D2_10TH_ORDER_5 = 8/25200

    # Finite differences matrices - second derivatives to second, fourth,
    # sixth and eighth order:

    if order == 2:
        grad2_2 = np.zeros((Npts, Npts))
        grad2_2 += np.diag(np.full(Npts-1, D2_2ND_ORDER_1), -1)
        grad2_2 += np.diag(np.full(Npts-0, D2_2ND_ORDER_0), +0)
        grad2_2 += np.diag(np.full(Npts-1, D2_2ND_ORDER_1), +1)
        return grad2_2

    elif order == 4:
        grad2_4 = np.zeros((Npts, Npts))
        grad2_4 += np.diag(np.full(Npts-2, D2_4TH_ORDER_2), -2)
        grad2_4 += np.diag(np.full(Npts-1, D2_4TH_ORDER_1), -1)
        grad2_4 += np.diag(np.full(Npts-0, D2_4TH_ORDER_0), +0)
        grad2_4 += np.diag(np.full(Npts-1, D2_4TH_ORDER_1), +1)
        grad2_4 += np.diag(np.full(Npts-2, D2_4TH_ORDER_2), +2)
        return grad2_4

    elif order == 6:
        grad2_6 = np.zeros((Npts, Npts))
        grad2_6 += np.diag(np.full(Npts-3, D2_6TH_ORDER_3), -3)
        grad2_6 += np.diag(np.full(Npts-2, D2_6TH_ORDER_2), -2)
        grad2_6 += np.diag(np.full(Npts-1, D2_6TH_ORDER_1), -1)
        grad2_6 += np.diag(np.full(Npts-0, D2_6TH_ORDER_0), +0)
        grad2_6 += np.diag(np.full(Npts-1, D2_6TH_ORDER_1), +1)
        grad2_6 += np.diag(np.full(Npts-2, D2_6TH_ORDER_2), +2)
        grad2_6 += np.diag(np.full(Npts-3, D2_6TH_ORDER_3), +3)
        return grad2_6

    elif order == 8:
        grad2_8 = np.zeros((Npts, Npts))
        grad2_8 += np.diag(np.full(Npts-4, D2_8TH_ORDER_4), -4)
        grad2_8 += np.diag(np.full(Npts-3, D2_8TH_ORDER_3), -3)
        grad2_8 += np.diag(np.full(Npts-2, D2_8TH_ORDER_2), -2)
        grad2_8 += np.diag(np.full(Npts-1, D2_8TH_ORDER_1), -1)
        grad2_8 += np.diag(np.full(Npts-0, D2_8TH_ORDER_0), +0)
        grad2_8 += np.diag(np.full(Npts-1, D2_8TH_ORDER_1), +1)
        grad2_8 += np.diag(np.full(Npts-2, D2_8TH_ORDER_2), +2)
        grad2_8 += np.diag(np.full(Npts-3, D2_8TH_ORDER_3), +3)
        grad2_8 += np.diag(np.full(Npts-4, D2_8TH_ORDER_4), +4)
        return grad2_8

    elif order == 10:
        grad2_10 = np.zeros((Npts, Npts))
        grad2_10 += np.diag(np.full(Npts-5, D2_10TH_ORDER_5), -5)
        grad2_10 += np.diag(np.full(Npts-4, D2_10TH_ORDER_4), -4)
        grad2_10 += np.diag(np.full(Npts-3, D2_10TH_ORDER_3), -3)
        grad2_10 += np.diag(np.full(Npts-2, D2_10TH_ORDER_2), -2)
        grad2_10 += np.diag(np.full(Npts-1, D2_10TH_ORDER_1), -1)
        grad2_10 += np.diag(np.full(Npts-0, D2_10TH_ORDER_0), +0)
        grad2_10 += np.diag(np.full(Npts-1, D2_10TH_ORDER_1), +1)
        grad2_10 += np.diag(np.full(Npts-2, D2_10TH_ORDER_2), +2)
        grad2_10 += np.diag(np.full(Npts-3, D2_10TH_ORDER_3), +3)
        grad2_10 += np.diag(np.full(Npts-4, D2_10TH_ORDER_4), +4)
        grad2_10 += np.diag(np.full(Npts-5, D2_10TH_ORDER_5), +5)
        return grad2_10

    else:
        raise ValueError(order)


from FEDVR import FiniteElements1D

# Total spatial region:
x_min = -10
x_max = 10

orders = [2, 4, 6, 8, 10]
n_elements = 20

errors_fedvr = []
errors_fd = defaultdict(list)

mean_coupled_points_fedvr = []
mean_coupled_points_fd = defaultdict(list)

N_arr = range(3, 10)


def V(x):
    return x**2

# Compute high accuracy energy for comparison:
Npts = 1024
x_fd = np.linspace(x_min, x_max, Npts)
dx_fd = x_fd[1] - x_fd[0]
D2_fd = make_fd_operator(Npts, order=10)/dx_fd**2
D2_fd[0, :] = D2_fd[-1, :] = D2_fd[:, 0] = D2_fd[:, -1] = 0
V_fd = 0.5 * np.diag(V(x_fd))
H_fd = -0.5 * D2_fd + V_fd
evals_fd, evecs_fd = np.linalg.eigh(H_fd)
exact_E = min(evals_fd)
print(f'exact: {exact_E}')

for N in N_arr:
    # how many other points are edge points coupled to?
    edge_coupled_points = 2 * (N - 1)
    interior_coupled_points = (N - 2)
    mean_coupled_points = (edge_coupled_points + (N - 2) * interior_coupled_points) / (N - 1)

    mean_coupled_points_fedvr.append(mean_coupled_points)
    Npts = (N -1) * n_elements + 1
    elements = FiniteElements1D(N, n_elements, x_min, x_max)

    def make_total_operator(operator):
        total_operator = np.zeros((N*n_elements - n_elements + 1, N*n_elements - n_elements + 1))
        for i in range(n_elements):
            start = i*N - i
            end = i*N - i + N
            total_operator[start:end, start:end] += operator
        total_operator[0, :] = total_operator[-1, :] = total_operator[:, 0] = total_operator[:, -1] = 0
        return total_operator

    D2_single_element = elements.second_derivative_operator()
    D2_fedvr = make_total_operator(D2_single_element)
    x_fedvr = np.array(list(elements.points[:, :-1].flatten()) + [x_max])
    V_fedvr = 0.5 * np.diag(V(x_fedvr))
    H_fedvr = -0.5 * D2_fedvr + V_fedvr
    evals_fedvr, evecs_fedvr = np.linalg.eigh(H_fedvr)

    print(f"N = {N}, Npts = {Npts} E = {min(evals_fedvr)}")
    error_fedvr = abs((min(evals_fedvr) - exact_E) / exact_E)

    errors_fedvr.append(error_fedvr)
  
    for order in orders:
        mean_coupled_points_fd[order].append(order)

        x_fd = np.linspace(x_min, x_max, Npts)
        dx_fd = x_fd[1] - x_fd[0]
        D2_fd = make_fd_operator(Npts, order)/dx_fd**2
        D2_fd[0, :] = D2_fd[-1, :] = D2_fd[:, 0] = D2_fd[:, -1] = 0
        V_fd = 0.5 * np.diag(V(x_fd))
        H_fd = -0.5 * D2_fd + V_fd
        evals_fd, evecs_fd = np.linalg.eigh(H_fd)
        print(f"  FD{order} E = {min(evals_fd)}")
        error_fd = abs((min(evals_fd) - exact_E) / exact_E)
        errors_fd[order].append(error_fd)



FIG_WIDTH = 3.25
FIG_HEIGHT = 2.75

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

plt.semilogy(N_arr, errors_fedvr, 'o-', label=R'$\textsc{fedvr}$', marker='s', zorder=10)
for order in orders:
    plt.semilogy(N_arr, errors_fd[order], 'o-', label=R'$\textsc{fd}_{%s}$ equiv.' % str(order))

equiv_err_lower_bound = []
equiv_err_upper_bound = []

for i, mean_coupled_points in enumerate(mean_coupled_points_fedvr):
    # Round down to a multiple of two:
    lower_bound_order = int(round(mean_coupled_points - mean_coupled_points % 2))
    upper_bound_order = lower_bound_order + 2

    # The errors of the two finite difference schemes with these orders
    equiv_err_lower_bound.append(errors_fd[lower_bound_order][i])
    equiv_err_upper_bound.append(errors_fd[upper_bound_order][i])

plt.fill_between(N_arr, equiv_err_lower_bound, equiv_err_upper_bound, facecolor='grey', alpha=0.5)

# plt.plot(N_arr, mean_coupled_points_fedvr, 'o-')
# for order in orders:
#     plt.plot(N_arr, mean_coupled_points_fd[order], 'o-')




# gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1,
#                        right=0.8, top=0.9, wspace=0.2, hspace=0.075)


# plt.plot(np.array(Ns_fedvr) - 2, np.array(max_evals_fedvr)*dx_av**2, marker='o', linestyle='-',
#              color=colors['blue'], markeredgecolor=colors['blue'], label=R'$\textsc{fedvr}$')

# plt.plot(Ns_FD, max_evals_fd, marker='s', linestyle='-',
#              color=colors['orange'], markeredgecolor=colors['orange'],
#              label='Finite differences')


# plt.axhline(np.pi**2, label='Fourier limit', linestyle='--', color='orange')

plt.legend(loc='lower left')

plt.xlabel(R'$\textsc{dvr}$ points per element')
plt.ylabel(R'Fractional error in $E_0$')
# plt.legend(loc='upper left')
plt.savefig('../fedvr_vs_fd_harmonic_groundstate.pdf')