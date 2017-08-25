import numpy as np
from FEDVR import Element
pi = np.pi


N = 20
element = Element(N, left_edge=-1, right_edge=1, N_left=N, N_right=N, width_left=2, width_right=2)
element.weights[0] *= 2
element.weights[-1] *= 2
x = np.linspace(-1, 1, 1000)


vec = np.random.rand(N)*np.sqrt(element.weights)

dx_max = np.diff(element.points).max()
dx_min = np.diff(element.points).min()

k = 1.5*np.pi/dx_max


def f(x):
    return np.sin(k*x)


vec = element.make_vector(f)

vec_interp = element.interpolate_vector(vec, x).real

import matplotlib.pyplot as plt
plt.plot(x, f(x))
plt.plot(x, vec_interp)
plt.plot(element.points, vec.real/np.sqrt(element.weights), 'ko')


plt.show()
