import os
import csv
import numpy as np
import pandas as pd
from openap import aero, prop, Thrust, Drag

np.set_printoptions(suppress=True)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

root = os.path.dirname(os.path.realpath(__file__))

nX = 11
nY = 9

stdn1 = np.array([
    3, 3, 4,            # NACp 11
    0.3, 0.3, 0.46,     # NACv 4
    0.4, 0.4, 0.2       # emeperic
]) / 2

stdn2 = np.array([
    10, 10, 15,         # NACp 10
    1, 1, 1.52,         # NACv 3
    1.6, 1.6, 0.6       # emeperic
]) / 2

stdn3 = np.array([
    30, 30, 45,         # NACp 9
    3, 3, 4.57,         # NACv 2
    5.0, 5.0, 2         # emeperic
]) / 2

stdn4 = np.array([
    93, 93, 135,        # NACp 8
    10, 10, 15.24,      # NACv 1
    15, 15, 6           # emeperic
]) / 2

stdn0 = stdn1 / 2

stdns = {'n0':stdn0, 'n1':stdn1, 'n2':stdn2, 'n3':stdn3, 'n4':stdn4}

# kernal paramter
kpm = 0.005
kpe = 0.005
# kpe = 0.01
kpsi = np.radians(2)

# paramter for the auto-regressive model for vz, vw
# alpha_vz, sigma_vz = (0.9997, 0.1423)
alpha_vz, sigma_vz = (0.9997, 0.1423*2)
alpha_vwx, sigma_vwx = (1.0003, 0.0733)
alpha_vwy, sigma_vwy = (1.0003, 0.0842)
alpha_tau, sigma_tau = (1.0000, 0.1223)


def X2Y(X):
    nrow, ncol = X.shape

    if ncol == 1:
        m, eta, x, y, z, vax, vay, vz, vwx, vwy, tau = X
        vgx = vax + vwx
        vgy = vay + vwy
        Y = np.array([x, y, z, vgx, vgy, vz, vwx, vwy, tau])
    else:
        m, eta, x, y, z, vax, vay, vz, vwx, vwy, tau = np.split(X, ncol, 1)
        vgx = vax + vwx
        vgy = vay + vwy
        Y = np.hstack([x, y, z, vgx, vgy, vz, vwx, vwy, tau])

    return Y

def Y2X(Y):
    x, y, z, vgx, vgy, vz, vwx, vwy, tau = Y
    vax =  vgx - vwx
    vay =  vgy - vwy
    X = [None, None, x, y, z, vax, vay, vz, vwx, vwy, tau]
    return X



def printarray(arr, label=None):
    print(label, ':\t', end=' ')
    for a in arr:
        if isinstance(a, str):
            print(a + '\t', end=' ')
        else:
            if a > 1000:
                print('%d \t' % a, end=' ')
            elif a > 100:
                print('%.1f \t' % a, end=' ')
            elif a < 0:
                print('%.1f \t' % a, end=' ')
            elif a < 3:
                print('%.3f \t' % a, end=' ')
            else:
                print('%.2f \t' % a, end=' ')
    print()


class SIR():
    def __init__(self, **kwargs):
        self.ac = kwargs.get('ac')
        self.eng = kwargs.get('eng')
        self.time = kwargs.get('time')
        self.Y = kwargs.get('obs')

        # slightly increase the cov matrix (factor of 0.2), for better convergency
        self.noise = kwargs.get('noise')
        self.stdn = stdns[self.noise] * 1.2

        self.R = np.zeros((nY, nY))
        np.fill_diagonal(self.R, self.stdn**2)

        self.thrust = Thrust(self.ac, self.eng)
        self.drag = Drag(self.ac)

        aircraft = prop.aircraft(self.ac)
        self.mmin = aircraft['limits']['OEW']
        self.mmax = aircraft['limits']['MTOW']
        self.mrange = (self.mmin, self.mmax)

        self.eta_min = kwargs.get('eta_min', 0.80)
        self.eta_max = kwargs.get('eta_max', 1)

        self.kstd_m = kpm * (self.mmax - self.mmin)
        self.kstd_eta = kpe * (1 - self.eta_min)

        self.X = None
        self.nP = None
        self.now = 0
        self.neff = None
        self.X_true = None

        logfn = kwargs.get('logfn', None)

        self.xlabels = [
            'm (kg)', '$\eta$ (-)',
            'x (m)', 'y (m)', 'z (m)',
            '$v_{ax}$ (m/s)', '$v_{ay}$ (m/s)',
            '$v_z$ (m/s)',
            '$v_{wx}$ (m/s)', '$v_{wy}$ (m/s)', '$\\tau$ (K)'
        ]

        self.ylabels = [
            'x', 'y', 'z',
            '$v_{gx}$', '$v_{gy}$',
            '$v_z$',
            '$v_{wx}$', '$v_{wy}$', '$\\tau$'
        ]

        if (logfn is not None):
            if ('.log' not in logfn):
                raise RuntimeError('Log file must end with .log')

            self.log = root + '/smclog/' + logfn
            print('writing to log:', self.log)

            header = ['time'] + self.ylabels \
                    + [l+" avg" for l in self.xlabels] \
                    + [l+" med" for l in self.xlabels] \
                    + [l+" min" for l in self.xlabels] \
                    + [l+" max" for l in self.xlabels]

            with open(self.log, 'wt') as fcsv:
                writer = csv.writer(fcsv, delimiter=',')
                writer.writerow(header)

        else:
            self.log = None


    def state_update(self, X, dt):
        nrow, ncol = X.shape

        m, eta, x, y, z, vax, vay, vz, vwx, vwy, tau = np.split(X, X.shape[1], 1)

        vgx = vax + vwx
        vgy = vay + vwy

        vg = np.sqrt(vgx**2 + vgy**2)
        va = np.sqrt(vax**2 + vay**2)

        psi = np.arctan2(vax, vay)

        gamma = np.arcsin(vz/va)

        T = self.thrust.climb(va/aero.kts, z/aero.ft, vz/aero.fpm)
        D = self.drag.clean(m, va/aero.kts, z/aero.ft)

        a = (eta * T - D) / m - aero.g0 * np.sin(gamma)
        print(np.mean(a))

        m1 = m
        eta1 = eta

        x1 = x + vgx * dt
        y1 = y + vgy * dt
        z1 = z + vz * dt

        va1 = va + a * dt
        # va1 = va + a * np.cos(gamma) * dt
        vax1 = va1 * np.sin(psi)
        vay1 = va1 * np.cos(psi)

        evz = np.random.normal(0, sigma_vz, nrow)
        vz1 = alpha_vz * vz + evz.reshape(-1, 1) * dt
        # vz1 = alpha_vz * vz + a * np.sin(gamma) * dt + evz.reshape(-1, 1) * dt

        evwx = np.random.normal(0, sigma_vwx, nrow)
        vwx1 = alpha_vwx * vwx + evwx.reshape(-1, 1) * dt

        evwy = np.random.normal(0, sigma_vwy, nrow)
        vwy1 = alpha_vwy * vwy + evwy.reshape(-1, 1) * dt

        etau = np.random.normal(0, sigma_tau, nrow)
        tau1 = alpha_tau * tau + etau.reshape(-1, 1) * dt

        X = np.hstack([m1, eta1, x1, y1, z1, vax1, vay1, vz1, vwx1, vwy1, tau1])

        return X


    def logsm(self):
        if self.log is None:
            return

        t = self.now

        if t in self.time:
            idx = list(self.time).index(t)
            measurement = self.Y[:, idx]
        else:
            measurement = np.ones(self.Y.shape[0]) * np.nan

        state_avgs = np.average(self.X, weights=self.W.T, axis=0)
        state_meds = np.median(self.X, axis=0)
        # state_mins = np.percentile(self.X, 2.5, axis=0)
        # state_maxs = np.percentile(self.X, 95.5, axis=0)
        state_mins = np.min(self.X, axis=0)
        state_maxs = np.max(self.X, axis=0)

        row = np.hstack([[t], measurement, state_avgs, state_meds, state_mins, state_maxs])

        with open(self.log, 'at') as fcsv:
            writer = csv.writer(fcsv, delimiter=',')
            writer.writerow(row)


    def compute_neff(self):
        self.neff = 1 / np.sum(np.square(self.W))


    def printstates(self, t):
        # ---- Debug ----
        obsv = Y2X(self.Y[:, t])
        obsv[0:2] = ['*****', '****']
        avgs = np.average(self.X, weights=self.W, axis=0)
        meds = np.median(self.X, axis=0)
        # mins = np.percentile(self.X, 2.5, axis=0)
        # maxs = np.percentile(self.X, 95.5, axis=0)
        mins = np.min(self.X, axis=0)
        maxs = np.max(self.X, axis=0)

        printarray(obsv, 'obsv')
        printarray(avgs, 'avgs')
        # printarray(meds, 'meds')
        printarray(mins, 'mins')
        printarray(maxs, 'maxs')


    def pickle_particles(self, fname):
        import os, pickle
        root = os.path.dirname(os.path.realpath(__file__))
        fpkl = open(root+'/data/'+fname, 'wb')
        pickle.dump({'X': self.X, 'W': self.W}, fpkl)


    def init_particles(self, at=0, n_particles=50000):
        Mu0 = Y2X(self.Y[:, at])
        Mu0[0:2] = [0, 0]

        Var0 = np.zeros((nX, nX))
        np.fill_diagonal(
            Var0,
            (np.append([0, 0], self.stdn))**2
        )

        printarray(Mu0, 'Mu0')
        printarray(np.diag(Var0), 'Var0')

        self.X = np.random.multivariate_normal(Mu0, Var0, n_particles)
        self.nP = n_particles

        m_inits = np.random.uniform(self.mmin, self.mmax, n_particles)

        # mass-related initialization (recommended)
        er_mins = 1 - (1-self.eta_min) * (self.mmax - m_inits) / (self.mmax - self.mmin)
        eta_inits = np.random.uniform(er_mins, self.eta_max, n_particles)

        # # Uniform initialization
        # eta_inits = np.random.uniform(self.eta_min, self.eta_max, n_particles)

        self.X[:, 0] = m_inits
        self.X[:, 1] = eta_inits

        self.W = np.ones(n_particles) / (n_particles)
        return

    def resample(self):
        """
        References: J. S. Liu and R. Chen. Sequential Monte Carlo methods for dynamic
           systems. Journal of the American Statistical Association,
           93(443):1032–1044, 1998.
        """
        N = self.nP
        W = self.W
        idx = np.zeros(N, 'i')

        # take int(N*w) copies of each weight, which ensures particles with the
        # same weight are drawn uniformly
        copyn = (np.floor(N * W)).astype(int)

        a = np.where(copyn>0)[0]
        b = copyn[a]
        c = np.append(0, np.cumsum(b))
        for i, (i0, i1) in enumerate(zip(c[:-1], c[1:])):
            idx[i0:i1] = a[i]

        # use multinormal resample on the residual to fill up the rest. This
        # maximizes the variance of the samples
        k = c[-1]
        residual = W - copyn
        residual /= sum(residual)
        cumulative_sum = np.cumsum(residual)
        cumulative_sum[-1] = 1.     # avoid round-off errors: ensures sum is exactly one
        idx[k:N] = np.searchsorted(cumulative_sum, np.random.random(N-k))
        return idx

    def run(self, processdt=1, use_bada=True):
        if self.X is None:
            raise RuntimeError('Particles not initialized. Run SIR.init_particles() first.')

        self.now = self.time[0]
        self.logsm()

        for it, tm in enumerate(self.time):
            print('-' * 100)
            self.now = tm

            # ===== SIR update =====
            print("SIR / measurement update, time", self.now)
            self.printstates(it)

            # ---- weight update ----
            Y_true = X2Y(self.X)
            Yt = self.Y[:, it]
            Y = np.ones(Y_true.shape) * Yt
            DY = Y - Y_true
            Simga_inv = np.linalg.inv(self.R)
            self.W *= np.exp(-0.5 * np.einsum('ij,ij->i', np.dot(DY, Simga_inv), DY))

            self.W = np.where(self.X[:, 1] < self.eta_min, 0, self.W)
            self.W = np.where(self.X[:, 1] > self.eta_max, 0, self.W)
            self.W = np.where(self.X[:, 0] < self.mmin, 0, self.W)
            self.W = np.where(self.X[:, 0] > self.mmax, 0, self.W)

            self.W = self.W / np.sum(self.W)    # normalize

            # ---- resample ----
            idx = self.resample()
            self.X = self.X[idx, :]
            self.W = np.ones(self.nP) / self.nP

            if tm == self.time[-1]:
                break

            # ---- apply kernel ----
            print((bcolors.OKGREEN + "Kernel applied: [m:%d, eta:%f]" + bcolors.ENDC) % (self.kstd_m, self.kstd_eta))
            self.X[:, 0] = self.X[:, 0] + np.random.normal(0, self.kstd_m, self.nP)
            self.X[:, 1] = self.X[:, 1] + np.random.normal(0, self.kstd_eta, self.nP)

            epsi = np.random.normal(0, kpsi, self.nP)
            va = np.sqrt(self.X[:, 5]**2 + self.X[:, 6]**2)
            psi = np.arctan2(self.X[:, 5], self.X[:, 6])

            self.X[:, 5] = va * np.sin(psi+epsi)
            self.X[:, 6] = va * np.cos(psi+epsi)


            # ===== states evolution =====
            dtm = self.time[it+1] - self.time[it]
            dtp = min(processdt, dtm)             # process update time interval

            for j in range(int(dtm/dtp)):   # number of process update
                print("process update / state evolution, time", self.now)
                self.now = tm + (j+1) * dtp
                self.X = self.state_update(self.X, dtp)
                self.logsm()

        return
