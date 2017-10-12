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

colors_rgb = {'red': (200,80,80),
              'green': (80,200,130),
              'blue': (80,160,200),
              'orange': (240,170,50)}


# Convert to float colours:
colors = {name: [chan/255 for chan in value] for name, value in colors_rgb.items()}

import numpy as np

pi = np.pi

n = 13

nx = 1024

psi = np.zeros((n,n))
psi[2, 3] = 1

f_psi = np.fft.fft2(psi)

kx = 2*pi*np.fft.fftfreq(n)
ky = 2*pi*np.fft.fftfreq(n)

x_interp = np.linspace(0, n, nx, endpoint=False).reshape((1, nx))
y_interp = np.linspace(0, n, nx, endpoint=False).reshape((nx, 1))
psi_interp = np.zeros((nx, nx), dtype=complex)
for i in range(n):
    for j in range(n):
        psi_interp += (np.exp(1j*ky[i]*y_interp) *
                       np.exp(1j*kx[j]*x_interp) * f_psi[i, j])

FIG_WIDTH = 4.5
FIG_HEIGHT = 3.25

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
gs = gridspec.GridSpec(4, 1, left=0.075, bottom=0.11,
                      right=0.90, top=0.975, wspace=0.5, hspace=0.1)


dx = x_interp[0, 1] - x_interp[0, 0]
psi_norm = np.abs(psi_interp**2).sum()*dx*dx
psi_interp /= np.sqrt(psi_norm)
psimax = psi_interp.real.max()

ax1 = plt.subplot(gs[:3,:])

plt.imshow(psi_interp.real, extent=[0, n, 0, n], origin='lower',
           cmap=plt.get_cmap('seismic'),
           vmin=-1.0, vmax=1.0)
ax1.tick_params(direction='in')
ax1.xaxis.set_major_formatter(plt.NullFormatter())

for i in range(n+1):
    for j in range(n+1):
        plt.plot([i], [j], 'ko', markersize=1)

# for i in range(n):
#     plt.axhline(i+0.5, color='k', linewidth=1, linestyle='--')
#     plt.axvline(i+0.5, color='k', linewidth=1, linestyle='--')

plt.ylabel('$y$')
colorbar = plt.colorbar(label=R'$\phi_{32}(x, y)$')
colorbar.ax.tick_params(direction='in')
colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])

# Draw so we can inspect the size of the first plot:
plt.savefig('../basis_vecs.pdf', dpi=300)

ax2 = plt.subplot(gs[3,:])
plt.plot(x_interp.flatten(), psi_interp[int(round(1024*2/n))].real, color=colors['red'])
plt.grid(True, linestyle=':')
plt.axis([0, n, -0.3, 1.3])
plt.xlabel('$x$')
plt.ylabel('$\phi_{32}(x, 2)$')

ax1_bbox = ax1.get_position()
ax2_bbox = ax2.get_position()
ax2.set_position([ax1_bbox.x0, ax2_bbox.y0, ax1_bbox.width, ax2_bbox.height])
matplotlib.pyplot.xticks(range(1, n+1))
pts = np.zeros(n+1)
pts[3] = 1
plt.plot(range(n+1), pts, 'ko', markersize=1)

plt.savefig('../basis_vecs.pdf', dpi=300)
