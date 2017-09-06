import numpy as np
import h5py
import matplotlib.pyplot as plt

fss2_fig = plt.figure()
fss4_fig = plt.figure()
rk4_fig = plt.figure()

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