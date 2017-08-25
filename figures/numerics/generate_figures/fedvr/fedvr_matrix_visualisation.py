from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from FEDVR import FiniteElements1D

# Total spatial region over all MPI processes:
x_min = -5
x_max = 5

# total number of points:
Npts = 721

# Number of DVR basis functions per element:
for i in range(1, 17):

    if (Npts - 1) % i:
        # Skip if we can't fit an integer number of elements in the space:
        continue

    N = i + 1

    # Finite elements:
    n_elements = (Npts - 1) // i
    print(i, n_elements)
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

    vals, vecs = np.linalg.eig(D2_total)

    plt.plot(sorted(vals))

    # D2_total = make_total_operator(D2)
    # D2_total[D2_total == 0] = np.nan


    # plt.matshow(np.log(np.abs(D2_total)))
    # plt.show()

plt.show()