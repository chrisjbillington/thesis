from __future__ import division
import numpy as np

from matplotlib import rcParams
import matplotlib
matplotlib.use('PDF')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

pi = np.pi
mu = 1.26804880139e-30 # Chemical potential of all simulations initial conditions
g = 5.07219520555e-51  # nonlinear constant

FIG_WIDTH = 4.5
FIG_HEIGHT = 5.25

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

gs = gridspec.GridSpec(8, 20, left=0.12, bottom=0.00,
                       right=0.97, top=0.95, wspace=0.2, hspace=0.075)

timesteps = ['0.25', '0.5', '1.0']

ylim = [3e-15, 1e-5]

labels = {'0.25': R'$\Delta t = \tau_{\rm d} / 4$',
         '0.5': R'$\Delta t = \tau_{\rm d} / 2$',
         '1.0': R'$\Delta t = \tau_{\rm d}$'}

with h5py.File('stability_data.h5', 'r') as f:

    # Plot the error for TDSE_FSS2:
    ax = plt.subplot(gs[0:2, 0:9])
    template = 'evolution_{}tau_FSS2_TDSE'
    for timestep in timesteps:
        dataset = f[template.format(timestep)]
        errors = np.abs(dataset['step err'])
        plt.semilogy(dataset['time']*1e3, errors, label=labels[timestep])

    ax.set_ylim(ylim)
    ax.set_xlim([0, 200])
    ax.tick_params(direction="in")
    plt.xticks([50,100, 150], ['']*3)
    plt.yticks([1e-12, 1e-9, 1e-6])
    ax.xaxis.set_label_position("top")
    plt.xlabel(R'Schr${\rm \"o}$dinger wave equation')
    plt.legend(loc='lower right', ncol=2, columnspacing=1)

    # Plot the error for TDSE_FSS4:
    ax = plt.subplot(gs[2:4, 0:9])
    template = 'evolution_{}tau_FSS4_TDSE'
    for timestep in timesteps:
        dataset = f[template.format(timestep)]
        errors = np.abs(dataset['step err'])
        plt.semilogy(dataset['time']*1e3, errors, label=labels[timestep])

    ax.set_ylim(ylim)
    ax.set_xlim([0, 200])
    ax.tick_params(direction="in")
    plt.xticks([50,100, 150], ['']*3)
    plt.yticks([1e-12, 1e-9, 1e-6])
    plt.legend(loc='upper right', ncol=2, columnspacing=1)

     # Plot the error for TDSE_RK4_FD6:
    ax = plt.subplot(gs[4:6, 0:9])
    template = 'evolution_{}tau_RK4_FD6_TDSE'
    for timestep in timesteps:
        dataset = f[template.format(timestep)]
        errors = np.abs(dataset['step err'])
        plt.semilogy(dataset['time']*1e3, errors, label=labels[timestep])

    ax.set_ylim(ylim)
    ax.set_xlim([0, 200])
    ax.tick_params(direction="in")
    plt.xticks([50,100, 150])
    plt.yticks([1e-12, 1e-9, 1e-6])
    plt.legend(loc='upper right', ncol=2, columnspacing=1)

    # Plot the error for GPE_FSS2:
    ax = plt.subplot(gs[0:2, 9:18])
    template = 'evolution_{}tau_FSS2_GPE'
    for timestep in timesteps:
        dataset = f[template.format(timestep)]
        errors = np.abs(dataset['step err'])
        plt.semilogy(dataset['time']*1e3, errors, label=labels[timestep])

    ax.set_ylim(ylim)
    ax.set_xlim([0, 200])
    ax.tick_params(direction="in")
    plt.xticks([50,100, 150], ['']*3)
    plt.yticks([1e-12, 1e-9, 1e-6], ['']*3)
    ax.yaxis.set_label_position("right")
    plt.ylabel(R'$\textsc{fss2}$')
    ax.xaxis.set_label_position("top")
    plt.xlabel(R'Gross--Pitaevskii equation')
    plt.legend(loc='lower right', ncol=2, columnspacing=1)

    # Plot the error for GPE_FSS4:
    ax = plt.subplot(gs[2:4, 9:18])
    template = 'evolution_{}tau_FSS4_GPE'
    for timestep in timesteps:
        dataset = f[template.format(timestep)]
        errors = np.abs(dataset['step err'])
        plt.semilogy(dataset['time']*1e3, errors, label=labels[timestep])

    ax.set_ylim(ylim)
    ax.set_xlim([0, 200])
    ax.tick_params(direction="in")
    plt.xticks([50,100, 150], ['']*3)
    plt.yticks([1e-12, 1e-9, 1e-6], ['']*3)
    ax.yaxis.set_label_position("right")
    plt.ylabel(R'$\textsc{fss4}$')
    plt.legend(loc='upper right', ncol=2, columnspacing=1)

     # Plot the error for GPE_RK4_FD6:
    ax = plt.subplot(gs[4:6, 9:18])
    template = 'evolution_{}tau_RK4_FD6_GPE'
    for timestep in timesteps:
        dataset = f[template.format(timestep)]
        errors = np.abs(dataset['step err'])
        plt.semilogy(dataset['time']*1e3, errors, label=labels[timestep])

    ax.set_ylim(ylim)
    ax.set_xlim([0, 200])
    ax.tick_params(direction="in")
    plt.xticks([50,100, 150])
    plt.yticks([1e-12, 1e-9, 1e-6], ['']*3)
    ax.yaxis.set_label_position("right")
    plt.ylabel(R'$\textsc{rk4fd6}$')
    plt.legend(loc='upper right', ncol=2, columnspacing=1)

    # Plot the suimulation frames:
    psi_gpe = [f['GPE_0'][:], f['GPE_1478'][:], f['GPE_2956'][:]]
    psi_tdse = [f['TDSE_0'][:], f['TDSE_1478'][:], f['TDSE_2956'][:]]

    rho_gpe = [np.abs(psi**2) for psi in psi_gpe]
    rho_clip = max([psi.max() for psi in rho_gpe])
    rho_tdse = [np.abs(psi**2).clip(0, rho_clip) for psi in psi_tdse]

    frame_ax1 = plt.subplot(gs[7:8, 0:9])
    frame_ax1.set_xticks([])
    frame_ax1.set_yticks([])
    plt.imshow(np.concatenate(rho_tdse).transpose(), origin='lower',
               cmap=cm.gray, interpolation='none')

    frame_ax2 = plt.subplot(gs[7:8, 9:18])
    frame_ax2.set_xticks([])
    frame_ax2.set_yticks([])
    plt.imshow(np.concatenate(rho_gpe).transpose(),
               origin='lower', cmap=cm.gray, interpolation='none')


    # Colorbar:
    nx = psi_gpe[0].shape[1]
    colorbar_ax = plt.subplot(gs[7:8, 18:20])
    colormap = np.linspace(0, 1, nx)[:, np.newaxis] * np.ones(nx//10)
    plt.imshow(colormap, origin='lower', extent = [0, 0.1, 0, 1.0], cmap=cm.gray)
    colorbar_ax.set_xticks([])
    colorbar_ax.yaxis.tick_right()
    colorbar_ax.yaxis.set_ticks_position('both')
    colorbar_ax.yaxis.set_label_position("right")
    colorbar_ax.tick_params(direction="in")
    yticks = [0.2, 0.4, 0.6, 0.8]
    plt.yticks(yticks)
    colorbar_ax.yaxis.set_tick_params(size=1)
    plt.ylabel(R'$\frac g \mu \rho$')

    # # Move closer to frame axes:
    colorbar_ax.set_anchor('W')

    # Move all the axes up a little and label the frames with the times they
    # are at:
    for ax in [frame_ax1, frame_ax2, colorbar_ax]:
        
        pos = ax.get_position()
        new_pos = [pos.x0, pos.y0 + 0.04,  pos.width, pos.height] 
        ax.set_position(new_pos)
        x0 = pos.x0
        width = pos.width
        if ax is not colorbar_ax:
            fig.text(x0 + width/6, 0.02, R'$t=0$', ha='center')
            fig.text(x0 + width/2, 0.02, R'$t=20\, \mathrm{ms}$', ha='center')
            fig.text(x0 + 5*width/6, 0.02, R'$t=40\, \mathrm{ms}$', ha='center')


    # Shared axis labels:
    fig.text(0.5, 0.185, R'simulation time (ms)', ha='center')
    fig.text(0.00, 0.6, 'per-step error', va='center', rotation='vertical')

    # Work out the DPI of the raster images
    height_inches = frame_ax1.get_position().height*FIG_HEIGHT
    # assume nx = ny
    dpi = nx/height_inches
plt.savefig('../stability.pdf', dpi=dpi)