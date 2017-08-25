from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from FEDVR import FiniteElements1D

# Constants:
pi = np.pi
hbar = 1.054571726e-34
a_0  = 5.29177209e-11                       # Bohr radius
u    = 1.660539e-27                         # unified atomic mass unit
m  = 86.909180*u                            # 87Rb atomic mass
a  = 98.98*a_0                              # 87Rb |2,2> scattering length
g  = 4*pi*hbar**2*a/m                       # 87Rb self interaction constant
rho_max = 2.5e14*1e6                        # Desired maximum density
R = 7.5e-6      # Desired Thomas-Fermi radius
omega = np.sqrt(2*g*rho_max/(m*R**2))       # Trap frequency  corresponding to desired density and radius
mu = g*rho_max                              # Desired chemical potential of the groundstate
N_2D = pi*rho_max*R**2/2                    # Thomas Fermi estimate of atom number for given chemical potential.
healing_length = 1/np.sqrt(8*pi*a*rho_max)

# Total spatial region over all MPI processes:
x_min = -5
x_max = 5

# Number of DVR basis functions per element:
N = 10

# Finite elements:
n_elements = 5
elements = FiniteElements1D(N, n_elements, x_min, x_max)

x = elements.points

alpha = 30

N_interp = 10000
x_interp = np.linspace(x_min, x_max, N_interp*n_elements)

def plot_vector(vector, *args, **kwargs):
    x_interp, vec_interp = elements.interpolate_vector(vector, N_interp)
    plt.plot(x_interp, vec_interp, *args, **kwargs)
    for j in range(n_elements):
        for i, point in enumerate(elements.points[j, :]):
            if 0 < i < N-1:
                factor = np.sqrt(elements.weights[i])
            else:
                factor = np.sqrt(2*elements.weights[i])
            plt.plot([point], vector[j, i]/factor, 'ko')


def apply_operator(operator, vector):
    result = np.einsum('ij, xj -> xi', operator, vector)
    # Sum at edges:
    result[1:, 0] = result[:-1, -1] = result[1:, 0] + result[:-1, -1]
    return result


def make_total_operator(operator):
    total_operator = np.zeros((N*n_elements - n_elements + 1, N*n_elements - n_elements + 1))
    for i in range(n_elements):
        start = i*N - i
        end = i*N - i + N
        total_operator[start:end, start:end] += operator
    # total_operator[0, :] = total_operator[-1, :] = total_operator[:, 0] = total_operator[:, -1] = 0
    return total_operator


def make_total_vector(vector):
    total_vector = np.zeros((N*n_elements - n_elements + 1))
    for i in range(n_elements):
        start = i*N - i
        end = i*N - i + N
        total_vector[start:end] = vector[i]
    return total_vector


def decompose_total_vector(total_vector):
    vector = np.zeros((n_elements, N))
    for i in range(n_elements):
        start = i*N - i
        end = i*N - i + N
        vector[i] = total_vector[start:end]
    return vector


V_exact = x_interp**alpha
gradV_exact = alpha*x_interp**(alpha-1)
grad2V_exact = alpha*(alpha-1)*x_interp**(alpha-2)

V_fedvr_exact = elements.make_vector(lambda x: x**alpha).real
gradV_fedvr_exact = elements.make_vector(lambda x: alpha*x**(alpha-1)).real
grad2V_fedvr_exact = elements.make_vector(lambda x: alpha*(alpha-1)*x**(alpha-2)).real

D = elements.derivative_operator()
D2 = elements.second_derivative_operator()

gradV_fedvr_derived = apply_operator(D, V_fedvr_exact)
grad2V_fedvr_derived = apply_operator(D2, V_fedvr_exact)

D2_total = make_total_operator(D2)
grad2V_fedvr_exact_total = make_total_vector(grad2V_fedvr_exact)


# Test if total operator works:
V_fedvr_exact_total = make_total_vector(V_fedvr_exact)
grad2V_fedvr_derived_total = np.dot(D2_total, V_fedvr_exact_total)
grad2V_fedvr_derived_2 = decompose_total_vector(grad2V_fedvr_derived_total)
assert np.allclose(grad2V_fedvr_derived_2, grad2V_fedvr_derived)
# Yep, it works!


V_fedvr_sanitised_total = np.linalg.lstsq(D2_total, grad2V_fedvr_exact_total)[0]
V_fedvr_sanitised = decompose_total_vector(V_fedvr_sanitised_total)

grad2V_fedvr_exact_verification = apply_operator(D2, V_fedvr_sanitised)
grad2V_fedvr_exact_verification2 = decompose_total_vector(np.dot(D2_total, V_fedvr_sanitised_total))

plot_vector(grad2V_fedvr_exact_verification)
plot_vector(grad2V_fedvr_exact_verification2)

plt.grid(True)
plt.show()
assert np.allclose(np.dot(D2_total, V_fedvr_sanitised_total), grad2V_fedvr_exact_total)
assert np.allclose(grad2V_fedvr_exact_verification, grad2V_fedvr_exact)

# V is only determined up to a linear function, because we've essentially
# integrated twice to find it. So let's find out what linear function it
# should be to coincide with the original V at the edges.
y0 = (V_fedvr_exact[2, 0] - V_fedvr_sanitised[2, 0])/np.sqrt(2*elements.weights[0])
y1 = (V_fedvr_exact[2, -1] - V_fedvr_sanitised[2, -1])/np.sqrt(2*elements.weights[-1])
x0 = elements.points[2, 0]
x1 = elements.points[2, -1]

linear_correction =  elements.make_vector(lambda x: y0 + (y1 - y0)/(x1 - x0)*(x - x0)).real
deriv_linear_correction =  apply_operator(D, linear_correction)

V_fedvr_sanitised += linear_correction

gradV_fedvr_sanitised = apply_operator(D, V_fedvr_sanitised)
grad2V_fedvr_sanitised = apply_operator(D2, V_fedvr_sanitised)

# straight_line = 2*x_interp + 1
# straight_line_fedvr = elements.make_vector(lambda x: 2*x + 1)
# deriv_straight_line = 2*np.ones(len(x_interp))
# deriv_straight_line_fedvr = apply_operator(D2, straight_line_fedvr)


# plt.plot(x_interp, straight_line, label='exact')
# # plt.plot(x_interp, deriv_straight_line, label='exact deriv')
# plot_vector(straight_line_fedvr, label='fedvr')
# plot_vector(deriv_straight_line_fedvr, label='fedvr deriv')
# plt.legend()
# plt.grid(True)
# plt.show()



plt.figure()
plt.title('V')
plt.plot(x_interp, V_exact, label='Exact')
plot_vector(V_fedvr_exact, label='FEDVR interpolation of exact')
plot_vector(V_fedvr_sanitised, label='FEDVR interpolation of sanitised')
plot_vector(linear_correction, label='linear correction')
plt.grid(True)
plt.legend()

plt.figure()
plt.title('gradV')
plt.plot(x_interp, gradV_exact, label='Exact')
plot_vector(gradV_fedvr_exact, label='FEDVR interpolation of exact')
plot_vector(gradV_fedvr_derived, label='FEDVR differentiation')
plot_vector(gradV_fedvr_sanitised, label='FEDVR interpolation of sanitised')
plot_vector(deriv_linear_correction, label='linear correction')
plt.grid(True)
plt.legend()

plt.figure()
plt.title('grad2V')
plt.plot(x_interp, grad2V_exact, label='Exact')
plot_vector(grad2V_fedvr_exact, label='FEDVR interpolation of exact')
plot_vector(grad2V_fedvr_derived, label='FEDVR differentiation')
plot_vector(grad2V_fedvr_sanitised, label='FEDVR interpolation of sanitised')
plt.grid(True)
plt.legend()

plt.show()






import IPython
IPython.embed()
