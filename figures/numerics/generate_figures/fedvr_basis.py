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

from FEDVR import FiniteElements1D
pi = np.pi


N = 10
n_elements = 2
x_max = 18
elements = FiniteElements1D(N, n_elements, 0, x_max)

FIG_WIDTH = 4.25
FIG_HEIGHT = 2.25

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
# gs = gridspec.GridSpec(1, 1, left=0.2, bottom=0.1,
#                        right=0.8, top=0.95, wspace=0.2, hspace=0.075)

# ax = plt.axes()
dx_av = x_max / (n_elements * (N - 1))

for i in range(n_elements):
    for j in range(N):
        psi = np.zeros((n_elements, N), dtype=complex)
        psi[i, j] = 1 # np.sqrt(element.weights[i])

        x_all, psi_interp = elements.interpolate_vector(psi, 10000)

        plt.plot(x_all/dx_av, psi_interp.real, color=colors['red'])

for i in range(n_elements):
    for j in range(N):
        weight = 1/np.sqrt(elements.weights[j])
        if j in [0, N-1]:
            weight /= np.sqrt(2)
        plt.plot((elements.element_edges[i] + elements.element.points[j])/dx_av, weight, 'ko')
        plt.plot((elements.element_edges[i] + elements.element.points[j])/dx_av, 0, 'ko')

plt.axvline(x_max/2, linestyle='--', color='k')

plt.axis([0, (N-1)*n_elements, -0.5, 3])

plt.xlabel('$x$')
plt.ylabel(R'$\phi_i(x)$')
plt.savefig('../fedvr_basis.pdf')

