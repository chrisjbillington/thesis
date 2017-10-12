import h5py
import matplotlib.image
import numpy as np
import matplotlib.cm as cm

with h5py.File('vortex_test_rk4ilip.h5/0.h5', 'r') as f:
    with h5py.File('vortex_test_rk4.h5/0.h5', 'r') as g:
        i = 0
        while True:
            print(i)
            psi_rk4ilip = f['/output/evolution/psi'][i]
            psi_rk4 = g['/output/evolution/psi'][i]
            ratio = psi_rk4ilip/psi_rk4
            phasediff = np.angle(ratio)
            # fractional_mod_diff = abs(ratio)

            phasediff = phasediff.transpose(0, 2, 1, 3, 4, 5).reshape((32*7, 32*7))
            # fractional_mod_diff = fractional_mod_diff.transpose(0, 2, 1, 3, 4, 5).reshape((32*7, 32*7))

            matplotlib.image.imsave('evolution/phasediff_%04d.png' % i, np.abs(phasediff),
                                    origin='lower', cmap = cm.gist_rainbow)
            # matplotlib.image.imsave('evolution/mod_diff_%04d.png' % i, fractional_mod_diff,
            #                         vmin=0, vmax=2, origin='lower', cmap = cm.gray)

            i += 1
