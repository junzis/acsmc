import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from smc import X2Y

root = os.path.dirname(os.path.realpath(__file__))

def trajectory(pf):
    nX = pf.Y.shape[0]
    Y = pf.Y.copy()


    plt.figure(figsize=(12, 8))

    plt.suptitle('Flight Trajectory')

    for i in range(nX):
        ax = plt.subplot(5, 2, i+1)

        ax.plot(pf.time, Y[i, :], '.', color='k', lw=1)

        if pf.X_true is not None:
            Y_true = X2Y(pf.X_true.T).T
            ax.plot(pf.time, Y_true[i, :], color='green', lw=1)

        ax.set_ylabel(pf.ylabels[i])
        ax.yaxis.set_label_position("left")
        ax.yaxis.set_ticks_position("right")
        if i < nX-1:
            ax.set_xticks([])
        else:
            ax.set_xlabel('time (s)')

    ax6 = plt.subplot(5, 2, 10)
    ax6.plot((1,1), (1,1), '.', color='k', label='observations')
    if pf.X_true is not None:
        ax6.plot((1,1), (1,1), color='green', label='true values')
    ax6.set_xlim(0, 0.5)
    ax6.set_xlim(0, 0.5)
    ax6.set_xticks([])
    ax6.set_yticks([])
    plt.legend(fontsize=12, loc='center left')


    plt.show()


def convergence_all(pf):

    df = pd.read_csv(pf.log)

    nX = pf.X.shape[1]


    plt.figure(figsize=(12, 8))

    vax = df.iloc[:, 4] - df.iloc[:, 7]
    vay = df.iloc[:, 5] - df.iloc[:, 8]

    for i in range(nX):
        if i == 7:  # vz
            nplot = 6
        elif i == 5: # vax
            nplot = 7
        elif i == 6: # vay
            nplot = 8
        else:
            nplot = i+1

        ax = plt.subplot(6, 2, nplot)

        if i==0:
            color = 'red'
        elif i==1:
            color = 'blue'
        else:
            color = 'k'

        ax.fill_between(df.time, df.iloc[:, i+32], df.iloc[:, i+43], alpha=0.2, color=color)

        ax.plot(df.time, df.iloc[:, i+10], color='k')    # weighted mean
        # ax.plot(df.time, df.iloc[:, i+19], color='b')   # medians

        # measurements
        # i in [0, 1] no measurements

        if i in [2, 3, 4, 7, 8, 9, 10]:  # x,y,z,vz,vwx,vwy,tau
            ax.plot(df.time, df.iloc[:, i-1], '.', color='k')
        elif i == 5: # vax
            ax.plot(df.time, vax, '.', color='k')
        elif i == 6: # vay
            ax.plot(df.time, vay, '.', color='k')

        # plot X_true if exist
        if pf.X_true is not None:
            ax.plot(pf.time, pf.X_true[i, :], color='green')

        ax.set_ylabel(pf.xlabels[i])
        ax.yaxis.set_label_position("left")
        ax.yaxis.set_ticks_position("right")
        ax.set_xlim([df.time.min(), df.time.max()])
        # ax.set_xlabel('time (s)')

    ax6 = plt.subplot(6, 2, 12)
    ax6.plot((1,1), (1,1), color='k', label='estimations')
    ax6.plot((1,1), (1,1), '.', color='k', label='observations')
    if pf.X_true is not None:
        ax6.plot((1,1), (1,1), color='green', label='true values')
    ax6.set_xlim(0, 0.5)
    ax6.set_ylim(0, 0.5)
    ax6.set_xticks([])
    ax6.set_yticks([])
    plt.legend(fontsize=12, loc='center left')

    plt.show()


def convergence_m_eta(pf):
    df = pd.read_csv(pf.log)

    plt.figure(figsize=(12, 2))

    # mass
    ax = plt.subplot(121)
    ax.fill_between(df.time, df.iloc[:, 32], df.iloc[:, 43], alpha=0.3, color='red')
    ax.plot(df.time, df.iloc[:, 10], color='k')
    if pf.X_true is not None:
        ax.plot(pf.time, pf.X_true[0, :], color='green')
    ax.set_title('noise $\mathbf{\Sigma}_{%s}$' % pf.noise)
    ax.set_ylabel(pf.xlabels[0])
    ax.set_xlim([df.time.min(), df.time.max()])
    ax.get_xaxis().set_tick_params(direction='in')
    ax.set_yticks([])

    # thrust setting
    ax = plt.subplot(122)
    ax.fill_between(df.time, df.iloc[:, 33], df.iloc[:, 44], alpha=0.3, color='blue')
    ax.plot(df.time, df.iloc[:, 11], color='k')
    if pf.X_true is not None:
        ax.plot(pf.time, pf.X_true[1, :], color='green')
    ax.set_title('noise $\mathbf{\Sigma}_{%s}$' % pf.noise)
    ax.set_ylabel(pf.xlabels[1])
    ax.set_xlim([df.time.min(), df.time.max()])
    ax.get_xaxis().set_tick_params(direction='in')
    ax.set_yticks([])

    plt.show()


def particle_hist(pf):
    plt.figure(figsize=(8, 2))

    plt.subplot(121)
    # sns.kdeplot(pf.X[:, 0], shade=True, color='red')
    plt.hist(pf.X[:, 0], bins=50, range=(pf.mmin, pf.mmax), normed=True, color='red')
    ymin, ymax = plt.gca().get_ylim()
    plt.text(pf.mmin, ymax/2, ' $\mathbf{\Sigma}_{%s}$' % pf.noise, ha='left', va='center')
    plt.xlim((pf.mmin, pf.mmax))
    plt.yticks([])
    plt.title('m (kg)')

    plt.subplot(122)
    # sns.kdeplot(pf.X[:, 1], shade=True, color='blue')
    plt.hist(pf.X[:, 1], bins=50, range=(0.85, 1), normed=True, color='blue')
    plt.xlim((0.85, 1.02))
    ymin, ymax = plt.gca().get_ylim()
    plt.text(0.85, ymax/2, ' $\mathbf{\Sigma}_{%s}$' % pf.noise, ha='left', va='center')
    plt.yticks([])
    plt.title('$\eta$ (-)')

    plt.show()


def mass_per_thrust(pf):

    dm = (pf.mmax - pf.mmin) / 20
    ms = np.round(pf.X[:, 0] / dm) * 20
    m_unique = np.unique(ms)

    plt.figure(figsize=(8, 2))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    for i, m in enumerate(sorted(m_unique)):
        mask = (ms == m)

        mean, std = stats.norm.fit(pf.X[:, 0][mask])
        x = np.linspace(mean-3*std, mean+3*std, 100)
        y = stats.norm.pdf(x, mean, std)
        y = y / max(y) * len(ms[mask])
        ax1.plot(x, y, lw=1)
        ax1.fill_between(x, y, alpha=0.5)
        ax1.set_title('m')
        ax1.set_yticks([])
        ax1.set_xlim([pf.mmin, pf.mmax])

        mean, std = stats.norm.fit(pf.X[:, 1][mask])
        std = min(0.01, std)
        x = np.linspace(mean-3*std, mean+3*std, 100)
        y = stats.norm.pdf(x, mean, std/2)
        y = y / max(y) * len(ms[mask])
        ax2.plot(x, y, lw=1)
        ax2.fill_between(x, y, alpha=0.5)
        ax2.set_title('$\eta$')
        ax2.set_yticks([])
        ax2.set_xlim([0.85, 1.02])

    ymin, ymax = ax1.get_ylim()
    ax1.text(pf.mmin, ymax/2, ' $\mathbf{\Sigma}_{%s}$' % pf.noise, ha='left', va='center')
    ymin, ymax = ax2.get_ylim()
    ax2.text(0.85, ymax/2, ' $\mathbf{\Sigma}_{%s}$' % pf.noise, ha='left', va='center')

    plt.show()
