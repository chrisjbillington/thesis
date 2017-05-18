def split_step2(t, psi, dt):
    """"Evolve psi in time from t to t + dt using a single step of the second
    order Fourier split-step method with timestep dt"""

    # First evolve using the potential term for half a timestep:
    psi *= np.exp(-1j/hbar * V_real * 0.5 * dt)

    # Then evolve using the kinetic term for a whole timestep, tranforming to
    # and from Fourier space where the kinetic term is diagonal:
    psi = ifft2(np.exp(-1j/hbar * K_fourier * dt) * fft2(psi))

    # Then evolve with the potential term again for half a timestep:
    psi *= np.exp(-1j/hbar * V_real * 0.5 * dt)

    return psi

def split_step4(t, psi, dt):
    """"Evolve psi in time from t to t + dt using a single step of the fourth
    order Fourier split-step method with timestep dt"""
    p = 1/(4 - 4**(1/3.0))

    # Five applications of second-order split-step using timesteps
    # of size p*dt, p*dt, (1 - 4*p)*dt, p*dt, p*dt
    for subdt in [p*dt, p*dt, (1 - 4*p)*dt, p*dt, p*dt]:
        psi = split_step2(t, psi, subdt)
        t += subdt
    return psi
