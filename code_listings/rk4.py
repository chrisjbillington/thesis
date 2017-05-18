def rk4(t, t_final, dt, psi, dpsi_dt):
    """Evolve the initial array psi_initial forward in time from time t to
    t_final according to the differential equation dpsi_dt using fourth order
    Runge-Kutta with timestep dt"""
    while t < t_final:
        k1 = dpsi_dt(t, psi)
        k2 = dpsi_dt(t + 0.5 * dt, psi + 0.5 * k1 * dt)
        k3 = dpsi_dt(t + 0.5 * dt, psi + 0.5 * k2 * dt)
        k4 = dpsi_dt(t + dt, psi + k3 * dt)

        psi[:] += dt/6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        t += dt

    return psi