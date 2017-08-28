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

# Total spatial region over all MPI processes:
x_min = -5
x_max = 5

# total number of points:
n_elements = 5

N = 10

dx_av = (x_max - x_min) / ((N-1)*n_elements + 1)

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
D2_total[D2_total == 0] = np.nan


FIG_WIDTH = 3.25
FIG_HEIGHT = 2.25

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
gs = gridspec.GridSpec(1, 10, left=0.1, bottom=0.1,
                       right=0.8, top=0.9, wspace=0.2, hspace=0.075)

ax1 = plt.subplot(gs[:,0:9])
im = ax1.matshow(np.log(np.abs(D2_total) * dx_av**2))
plt.xlabel('$j$')
plt.ylabel('$i$')

ax2 = plt.subplot(gs[:,9])
plt.colorbar(im, cax=ax2, label=R'$\log\left(\Delta x_{\rm av}^2|\delta^2_{ij}|\right)$')
plt.savefig('../fedvr_D2_operator.pdf')
