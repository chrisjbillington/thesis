from __future__ import division
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

from FEDVR import FiniteElements1D

# Total spatial region:
x_min = -5
x_max = 5

# total number of points:
Npts = 721

Ns_fedvr = []
max_evals_fedvr = []

dx_av = (x_max - x_min) / Npts
# Number of DVR basis functions per element:
for i in range(2, 13):

    if (Npts - 1) % i:
        # Skip if we can't fit an integer number of elements in the space:
        continue

    N = i + 1
    Ns_fedvr.append(N)

    # Finite elements:
    n_elements = (Npts - 1) // i
    print(i, n_elements)
    elements = FiniteElements1D(N, n_elements, x_min, x_max)

    def make_total_operator(operator):
        total_operator = np.zeros((N*n_elements - n_elements + 1, N*n_elements - n_elements + 1))
        for i in range(n_elements):
            start = i*N - i
            end = i*N - i + N
            total_operator[start:end, start:end] += operator
        # total_operator[0, :] = total_operator[-1, :] = total_operator[:, 0] = total_operator[:, -1] = 0
        return total_operator



    D = elements.derivative_operator()
    D2 = elements.second_derivative_operator()

    D2_total = make_total_operator(D2)

    vals, vecs = np.linalg.eig(D2_total)

    max_evals_fedvr.append(max(abs(vals)))


# Finite differences matrices - second derivatives to second, fourth,
# sixth and eighth order:

grad2_2 = np.zeros((Npts, Npts))
grad2_2 += np.diag(np.full(Npts-1, D2_2ND_ORDER_1), -1)
grad2_2 += np.diag(np.full(Npts-0, D2_2ND_ORDER_0), +0)
grad2_2 += np.diag(np.full(Npts-1, D2_2ND_ORDER_1), +1)

grad2_4 = np.zeros((Npts, Npts))
grad2_4 += np.diag(np.full(Npts-2, D2_4TH_ORDER_2), -2)
grad2_4 += np.diag(np.full(Npts-1, D2_4TH_ORDER_1), -1)
grad2_4 += np.diag(np.full(Npts-0, D2_4TH_ORDER_0), +0)
grad2_4 += np.diag(np.full(Npts-1, D2_4TH_ORDER_1), +1)
grad2_4 += np.diag(np.full(Npts-2, D2_4TH_ORDER_2), +2)

grad2_6 = np.zeros((Npts, Npts))
grad2_6 += np.diag(np.full(Npts-3, D2_6TH_ORDER_3), -3)
grad2_6 += np.diag(np.full(Npts-2, D2_6TH_ORDER_2), -2)
grad2_6 += np.diag(np.full(Npts-1, D2_6TH_ORDER_1), -1)
grad2_6 += np.diag(np.full(Npts-0, D2_6TH_ORDER_0), +0)
grad2_6 += np.diag(np.full(Npts-1, D2_6TH_ORDER_1), +1)
grad2_6 += np.diag(np.full(Npts-2, D2_6TH_ORDER_2), +2)
grad2_6 += np.diag(np.full(Npts-3, D2_6TH_ORDER_3), +3)

grad2_8 = np.zeros((Npts, Npts))
grad2_8 += np.diag(np.full(Npts-4, D2_8TH_ORDER_4), -4)
grad2_8 += np.diag(np.full(Npts-3, D2_8TH_ORDER_3), -3)
grad2_8 += np.diag(np.full(Npts-2, D2_8TH_ORDER_2), -2)
grad2_8 += np.diag(np.full(Npts-1, D2_8TH_ORDER_1), -1)
grad2_8 += np.diag(np.full(Npts-0, D2_8TH_ORDER_0), +0)
grad2_8 += np.diag(np.full(Npts-1, D2_8TH_ORDER_1), +1)
grad2_8 += np.diag(np.full(Npts-2, D2_8TH_ORDER_2), +2)
grad2_8 += np.diag(np.full(Npts-3, D2_8TH_ORDER_3), +3)
grad2_8 += np.diag(np.full(Npts-4, D2_8TH_ORDER_3), +4)

Ns_FD = [2, 4, 6, 8]
max_evals_fd = []

for operator in [grad2_2, grad2_4, grad2_6, grad2_8]:
    vals, vecs = np.linalg.eig(operator)
    max_evals_fd.append(max(abs(vals)))


FIG_WIDTH = 3.25
FIG_HEIGHT = 2.25

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1,
                       right=0.8, top=0.9, wspace=0.2, hspace=0.075)


plt.plot(np.array(Ns_fedvr) - 2, np.array(max_evals_fedvr)*dx_av**2, marker='o', linestyle='-',
             color=colors['blue'], markeredgecolor=colors['blue'], label=R'$\textsc{fedvr}$')

plt.plot(Ns_FD, max_evals_fd, marker='s', linestyle='-',
             color=colors['orange'], markeredgecolor=colors['orange'],
             label='Finite differences')


plt.axhline(np.pi**2, label='Fourier limit', linestyle='--', color='orange')

plt.xlabel(R'Order of accuracy in $\Delta x_{\rm av}$')
plt.ylabel(R'$\Delta x_{\rm av}^2 \times \rho\left(\delta^{2 (n)}\right)$')
plt.legend(loc='upper left')
plt.savefig('../fedvr_eigenvalue_scaling.pdf')