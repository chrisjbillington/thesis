from __future__ import division, print_function
import numpy as np
from scipy.special import gamma as gamma_func

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

FIG_WIDTH = 4.48
FIG_HEIGHT = 4.48

import matplotlib.pyplot as plt


pi = np.pi
hbar = 1.054572e-34
k_B = 1.3806488e-23
mu_B = 9.27401e-24               # Bohr magneton
gF = -0.50182670925              # Lande g-factor of F=1 groundstate of Rubidium 87
m_Rb = 86.909180526*1.660539e-27 # Rubidium 87 mass

dB_dz = 2.5

sigma = np.logspace(-11, -5, 1000)

a = np.abs(gF*mu_B/(m_Rb)*dB_dz) # Between components with delta_m = 1

tau_pos_a = 2.156*np.sqrt(sigma/a)
tau_pos_b = - 2j*m_Rb*sigma**2/hbar
tau_vel_a = 1.253*hbar/(sigma*m_Rb*a)
tau_vel_b = - 1j*hbar**3/(2*a**2*m_Rb**3*sigma**4)

gamma_pos_real = 1/tau_pos_a
gamma_pos_imag = - tau_pos_b.imag/tau_pos_a**2
gamma_vel_real = 1/tau_vel_a
gamma_vel_imag = - tau_vel_b.imag/tau_vel_a**2

gamma_total_real = np.sqrt(gamma_pos_real**2 + gamma_vel_real**2)
gamma_total_imag = (gamma_pos_imag**(-1) + gamma_vel_imag**(-1))**(-1)

t = np.linspace(0,5/np.abs(gamma_total_real).min(), 10000).reshape((10000,1))

x = 1/2*a*t**2
k = m_Rb/hbar*a*t
exact_overlap = np.exp(-1/(8*sigma**2)*x**2 - 1j/2*x*k - sigma**2/2*k**2)
exact_gamma = 1/(exact_overlap.sum(axis=0)*(t[1] - t[0]))

plt.figure(figsize=(FIG_WIDTH,FIG_HEIGHT))

plt.loglog(sigma, exact_gamma.real, 'k-', label=r'exact numeric (real)', linewidth=3)
plt.loglog(sigma, exact_gamma.imag, 'k--', label=r'exact numeric (imag)', linewidth=3)

plt.loglog(sigma, gamma_total_real, 'b-', label='approx analytic (real)')
plt.loglog(sigma, gamma_total_imag, 'b--', label='approx analytic (imag)')

plt.loglog(sigma, gamma_vel_real, 'g-', label='large $\sigma$ analytic (real)')
plt.loglog(sigma, gamma_vel_imag, 'g--', label='large $\sigma$ analytic (imag)')

plt.loglog(sigma, gamma_pos_real, 'r-', label='small $\sigma$ analytic (real)')
plt.loglog(sigma, gamma_pos_imag, 'r--', label='small $\sigma$ analytic (imag)')
plt.grid(True)
plt.legend(loc=(0.0125, 0.3))
plt.axis([1e-11,1e-5,1,1e8])
plt.xlabel('Wavepacket size $\sigma$ (m)')
plt.ylabel('Decoherence rate (s$^{-1}$)')
# plt.title('Example markovian decay rate')
plt.savefig('../decoherence_rate_example.pdf')

t = t[:, 0]
dt = t[1] - t[0]
lambda_th = sigma[568]

# print('T = ', (2*pi*hbar)**2 / (2 * pi * m_Rb * k_B * lambda_th ** 2))
# assert 0
exact_overlap = exact_overlap[:, 568]
exact_gamma = exact_gamma[568]
gamma_overlap = np.exp(-exact_gamma*t)
# average_overlap = np.exp(-t/average_scale)

convolved_overlap = np.zeros(len(t), dtype=complex)
for i, tau in enumerate(t):
    print(i)
    convolved_overlap[:-1-i] += exact_overlap[i+1:]*dt
convolved_overlap /= convolved_overlap[0]


plt.figure(figsize=(FIG_WIDTH,FIG_HEIGHT))
plt.plot(t*1e6, exact_overlap.real,'k-', label='exact (real)')
plt.plot(t*1e6, exact_overlap.imag, 'k--', label='exact (imag)')
plt.plot(t*1e6, convolved_overlap.real, 'b-', label='unknown start time (real)')
plt.plot(t*1e6, convolved_overlap.imag, 'b--', label='unknown start time (imag)')
plt.grid(True)
# pl.plot(t*1e6, average_overlap.real, 'b-')
# pl.plot(t*1e6, average_overlap.imag, 'b--')
plt.plot(t*1e6, gamma_overlap.real, 'g-', label='Markovian (real)')
plt.plot(t*1e6, gamma_overlap.imag, 'g--', label='Markovian (imag)')

# plt.title('example decoherence factor')
plt.xlabel(r't ($\upmu$s)')
plt.legend()
plt.axis([0, 100, -0.2, 1])

plt.savefig('../decoherence_factor_example.pdf')
