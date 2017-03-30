import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

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


import numpy as np

n = 7
x = np.linspace(-1, 1, n).reshape((n, 1))
y = x.reshape((1, n))
psi = np.exp(-(x**2 + y**2))

FIG_WIDTH = 4.5
FIG_HEIGHT = 2.25

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
gs = gridspec.GridSpec(1, 3, left=0.075, bottom=0.15,
                        right=0.95, top=0.975, wspace=0.7, hspace=0.075)

plt.subplot(gs[:,:2])
plt.gca().tick_params(direction='in')

plt.imshow(psi, origin='lower', extent=[0.5, 7.5, 0.5, 7.5])
for i in range(1, n):
    for j in range(1, n+1):
        plt.gca().arrow(i, j, 1, 0, head_width=0.05, head_length=0.1, fc='w', ec='w')
for j in range(1, n):
    plt.gca().arrow(n, j, -(n-1), 1, head_width=0.05, head_length=0.1, fc='w', ec='w')

plt.xlabel('$i_x$')
plt.ylabel('$i_y$')
matplotlib.pyplot.yticks([1, 2, 3, 4, 5, 6, 7])
matplotlib.pyplot.xticks([1, 2, 3, 4, 5, 6, 7])

plt.subplot(gs[:,2])
plt.imshow(psi.ravel().reshape((n**2, 1)), extent=[0, 1, n**2 + 0.5, 0.5])
matplotlib.pyplot.yticks([1, 7, 14, 21, 28, 35, 42, 49])
plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
plt.tick_params(axis='y', which='both', left='off', right='on',
                labelright='on', labelleft='off')
plt.ylabel(R"$i_{xy} = i_x + n_xi_y$")
plt.gca().yaxis.set_label_position("right")

arrow = matplotlib.patches.FancyArrowPatch((0.625, 0.55), (0.775, 0.55), arrowstyle='-|>',
                                           transform=fig.transFigure, mutation_scale=5.0,
                                           facecolor='k')
fig.patches.append(arrow)


plt.savefig('../vector_unravel.pdf', dpi=300)