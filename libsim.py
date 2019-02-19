import numpy as np
from openap import aero, prop, Thrust, Drag
import smc

def calc_min_eta(m, mdl):
    ac = prop.aircraft(mdl)
    mtow = ac['limits']['MTOW']
    oew = ac['limits']['OEW']
    return 1 - 0.15 * (mtow - m) / (mtow - oew)


def initX(m, eta, vg, hdg, vw, wdir):
    hdg = np.radians(hdg)
    vgx = vg * np.sin(hdg)
    vgy = vg * np.cos(hdg)

    wdir = np.radians(wdir)
    vwx = vw * np.sin(wdir)
    vwy = vw * np.cos(wdir)

    x = 0
    y = 0
    z = 150
    vz = 12

    tau = aero.temperature(z)

    X = [m, eta, x, y, z, vgx, vgy, vz, vwx, vwy, tau]
    return X


def state_update(X, dt, mdl, eng):
    nrow, ncol = X.shape

    m, eta, x, y, z, vax, vay, vz, vwx, vwy, tau  = X

    vgx = vax + vwx
    vgy = vay + vwy

    vg = np.sqrt(vgx**2 + vgy**2)
    va = np.sqrt(vax**2 + vay**2)

    psi = np.arctan2(vax, vay)

    gamma = np.arcsin(vz/va)

    thrust = Thrust(mdl, eng)
    T = thrust.climb(va/aero.kts, z/aero.ft, vz/aero.fpm)

    drag = Drag(mdl)
    D = drag.clean(m, va/aero.kts, z/aero.ft)

    a = (eta * T - D) / m - aero.g0 * np.sin(gamma)

    m1 = m
    eta1 = eta

    x1 = x + vgx * dt
    y1 = y + vgy * dt
    z1 = z + vz * dt

    va1 = va + a * dt
    vax1 = va1 * np.sin(psi)
    vay1 = va1 * np.cos(psi)

    vz1 = vz
    vwx1 = vwx
    vwy1 = vwy
    tau1 = aero.temperature(z)

    X = np.array([m1, eta1, x1, y1, z1, vax1, vay1, vz1, vwx1, vwy1, tau1])

    return X


def observe(X, stdn):
    nrow, ncol = X.shape

    if ncol == 1:
        m0, eta0, x0, y0, z0, vax0, vay0, vz0, vwx0, vwy0, tau0 = X
        n = 1
    else:
        m0, eta0, x0, y0, z0, vax0, vay0, vz0, vwx0, vwy0, tau0 = np.split(X, ncol, 1)
        n = (nrow, 1)

    sx, sy, sz, svgx, svgy, svz, svwx, svwy, stau = stdn

    vgx0 = vax0 + vwx0
    vgy0 = vay0 + vwy0

    x1 = x0 + np.random.normal(0, sx, n)
    y1 = y0 + np.random.normal(0, sy, n)
    z1 = z0 + np.random.normal(0, sz, n)
    vgx1 = vgx0  + np.random.normal(0, svgx, n)
    vgy1 = vgy0  + np.random.normal(0, svgy, n)
    vz1 = vz0 + np.random.normal(0, svz, n)
    vwx1 = vwx0 + np.random.normal(0, svwx, n)
    vwy1 = vwy0 + np.random.normal(0, svwy, n)
    # tau1 = tau0 + np.random.normal(0, stau, n)
    tau1 = aero.temperature(z1) + np.random.normal(0, stau, n)

    Y_obs = np.hstack([x1, y1, z1, vgx1, vgy1, vz1, vwx1, vwy1, tau1])

    return Y_obs


def sim_trjectory(n_run, mdl, eng, X0, stdn):
    X = np.zeros((smc.nX, n_run))
    Y_obs = np.zeros((smc.nY, n_run))

    X0 = np.array(X0).reshape(-1, 1)

    for t in range(n_run):
        X[:, t] = X0[:, 0]

        Y0_obs = observe(X0, stdn)
        Y_obs[:, t] = Y0_obs

        X0 = state_update(X0, 1, mdl, eng)

    return X, Y_obs
