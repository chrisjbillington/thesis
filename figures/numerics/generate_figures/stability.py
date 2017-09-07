from __future__ import division
import numpy as np

from matplotlib import rcParams
import matplotlib
# matplotlib.use('PDF')
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

import h5py

FIG_WIDTH = 3.25
FIG_HEIGHT = 2.25

gs = gridspec.GridSpec(1, 1, left=0.1, bottom=0.1,
                       right=0.8, top=0.9, wspace=0.2, hspace=0.075)

timesteps = ['0.25', '0.5', '1.0']


with h5py.File('stability_data.h5', 'r') as f:
    for name in f:
        dataset = f[name]

        if 'RK4' in name:
            plt.figure(rk4_fig.number)
        elif 'FSS2' in  name:
            plt.figure(fss2_fig.number)
        elif 'FSS4' in  name:
            plt.figure(fss4_fig.number)
        else:
            raise ValueError(name)

        plt.semilogy(dataset['time']*1e3, np.abs(dataset['step err']), label=name)

        plt.legend()    

plt.show()