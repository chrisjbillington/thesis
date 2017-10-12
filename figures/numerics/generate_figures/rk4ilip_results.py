from __future__ import print_function, division

import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np
import h5py

pi = np.pi
mu = 1.26804880139e-30 # Chemical potential of all simulations initial conditions
g = 5.07219520555e-51  # nonlinear constant

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

methods_to_colors = {'rk4': colors['red'], 'rk4ilip': colors['green'], 'rk4ip': colors['blue'], 'split-step': colors['orange']}
methods_to_markers = {'rk4': 'v', 'rk4ilip': '^', 'rk4ip': 'o', 'split-step': 's'}
methods_to_labels = {'rk4': '\sc rk\oldstylenums 4',
                     'rk4ilip': '\sc rk\oldstylenums 4ilip',
                     'rk4ip': '\sc rk\oldstylenums 4ip',
                     'split-step': '\sc fss'}

FIG_WIDTH = 6.27
FIG_HEIGHT = 7.38
CROSS_SECTION_PLOT_TICK_SPACE = 0.05

def make_figure(f):
    plot_row_number = 0
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    gs = gridspec.GridSpec(8, 106, left=0.08, bottom=0.045, right=0.985, top=0.99, wspace=0.075, hspace=0.075)
    err_axes = []
    cross_secion_axes = []
    frame_axes = []
    for group_name in ['turbulent_bec', 'rotating_bec']:
        if group_name == 'turbulent_bec':
            methods = ['rk4ip', 'rk4ilip', 'rk4', 'split-step']
            prefix = 't'
        elif group_name == 'rotating_bec':
            methods = ['rk4ilip', 'rk4']
            prefix = 'r'
        group = f[group_name]
        for alpha in [4, 8, 12, 16]:
            frames = group['{}{}_{}_{}frames'.format(prefix, alpha, 16, 'rk4ilip')][:]
            ax = plt.subplot(gs[plot_row_number, :28])
            err_axes.append(ax)

            # The step error plots:
            for method in methods:
                worst_steperr = []
                for timestep_factor in [1, 2, 4, 8, 16]:
                    name = '{}{}_{}_{}'.format(prefix, alpha, timestep_factor, method)
                    dataset = group[name]
                    step_err = dataset['step err'][:]
                    worst_steperr.append(step_err.max() if len(step_err) > 40 else np.nan)

                color = methods_to_colors[method]
                marker = methods_to_markers[method]
                label = methods_to_labels[method]
                plt.semilogy(range(1, 6), worst_steperr, marker=marker, color=color, markeredgecolor=color, label=label)
            plt.axis(xmin=0.5, xmax=5.5, ymin=1e-17, ymax=1)
            plt.yticks([1e-3, 1e-6, 1e-9, 1e-12, 1e-15])
            plt.legend(loc='upper right', ncol=2, columnspacing=0.3)
            plt.xticks(range(1, 6), ['']*5)

            rho_frames = np.abs(frames)**2
            nx = frames.shape[1]
            n_middle_strip = 0.9/10*nx
            i_middle = nx//2
            psi_0_density_cross_secion = rho_frames[0, i_middle-n_middle_strip/2:i_middle+n_middle_strip/2, :]
            psi_0_column_density = psi_0_density_cross_secion.mean(axis=0)
            normalised_cross_section = g / mu * psi_0_column_density

            # The potential/cross secion plot:
            gs_offset = 5
            ax = plt.subplot(gs[plot_row_number, 28 + gs_offset:44 + gs_offset])
            cross_secion_axes.append(ax)
            x = np.linspace(-1, 1, nx)
            V = (x/0.9)**alpha
            plt.plot(x, V, 'k-', label=R"$\left(\frac x R\right)^{%d}$"%alpha)
            plt.plot(x, normalised_cross_section, color=colors['red'], label=R"$\frac g \mu \tilde \rho_0(x)$")
            plt.axis([-1, 1, 0, 2])
            plt.xticks([-0.9, 0, 0.9], ['', '', ''])
            yticks = [0.5, 1.0, 1.5, 2.0]
            plt.yticks(yticks)
            plt.legend(loc='upper center')

            # Three frames from the simulation:
            rho = np.abs(frames)**2

            ax = plt.subplot(gs[plot_row_number, 44 + gs_offset:92 + gs_offset])
            frame_axes.append(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(np.concatenate(rho).transpose(), origin='lower', cmap=cm.gray, interpolation='none')

            # Colorbar:
            ax = plt.subplot(gs[plot_row_number, 92 + gs_offset:])
            colormap = np.linspace(0, 1, nx)[:, np.newaxis] * np.ones(nx/10)
            maxval = g/mu * rho_frames.max()
            plt.imshow(colormap, origin='lower', extent = [0, maxval/10, 0, maxval], cmap=cm.gray)
            ax.set_xticks([])
            ax.yaxis.tick_right()
            ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_label_position("right")
            yticks = [0.2, 0.4, 0.6, 0.8, 1.0]
            if maxval > 1.2:
                yticks += [1.2]
            plt.yticks(yticks)
            ax.yaxis.set_tick_params(size=1)
            plt.ylabel(R'$\frac g \mu \rho$')

            # Move closer to frame axes:
            ax.set_anchor('W')

            plot_row_number += 1

    # Label the shared x axis of the error plots:
    plt.sca(err_axes[-1])
    err_labels = [R'$\tau_{\mathrm{d}}$',
                  R'$\tau_{\mathrm{d}}\over 2$',
                  R'$\tau_{\mathrm{d}}\over 4$',
                  R'$\tau_{\mathrm{d}}\over 8$',
                  R'$\tau_{\mathrm{d}}\over {16}$']

    plt.xticks(range(1, 6), err_labels)

    # Label the y axes and the shared x axis for the error plots:
    # Get the y point halfway up the four plots and the x point halfway along the bottom:
    top_pos = err_axes[0].get_position()
    top_y = top_pos.y0 + top_pos.height

    bottom_pos = err_axes[-1].get_position()
    bottom_y = bottom_pos.y0
    middle_y = (bottom_y + top_y)/2
    err_middle_x = bottom_pos.x0 + 0.5*bottom_pos.width
    fig.text(err_middle_x, 0.01, 'integration timestep', ha='center')
    fig.text(0.00, middle_y, 'maximum step error', va='center', rotation='vertical')

    # Label the shared x axis of the cross section plots:
    plt.sca(cross_secion_axes[-1])
    cross_secion_labels = [R'$-R$', R'$0$', R'$R$']

    plt.xticks([-0.9, 0, 0.9], cross_secion_labels)

    # Get the x point halfway along the bottom:
    bottom_pos = cross_secion_axes[-1].get_position()
    cross_secion_middle_x = bottom_pos.x0 + 0.5*bottom_pos.width
    fig.text(cross_secion_middle_x, 0.01, R'$x$', ha='center')

    # Label the frames with the times they are at:
    frame_axis_pos = frame_axes[0].get_position()
    x0 = frame_axis_pos.x0
    width = frame_axis_pos.width
    fig.text(x0 + width/6, 0.03, R'$t=0$', ha='center')
    fig.text(x0 + width/2, 0.03, R'$t=20\, \mathrm{ms}$', ha='center')
    fig.text(x0 + 5*width/6, 0.03, R'$t=40\, \mathrm{ms}$', ha='center')


    # Work out what dpi to save at in order to get the frames at native
    # resolution:
    height_inches = frame_axes[-1].get_position().height*FIG_HEIGHT
    # assume nx = ny
    dpi = nx/height_inches
    return dpi

with h5py.File('rk4ilip_results.h5', 'r') as f:
    dpi = make_figure(f)
    plt.savefig('../rk4ilip_results.pdf', dpi=dpi)
