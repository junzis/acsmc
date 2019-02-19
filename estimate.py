import os, sys
import pandas as pd
import numpy as np
import argparse
from openap import aero
import smc, smc_vis


parser = argparse.ArgumentParser()
parser.add_argument('--ac', help="aircraft model", required=True)
parser.add_argument('--eng', help="aircraft engine type", required=True)
parser.add_argument('--fin', help="trajectory csv file", required=True)
parser.add_argument('--noise', choices=['n1','n2','n3','n4'], help="noise model", required=True)
args = parser.parse_args()

ac = args.ac
eng = args.eng
fin = args.fin
stdn = args.noise

ndata = 20

fn, ext = os.path.splitext(os.path.basename(fin))


def search_start(df, n):
    """find a segmention without turns"""
    for i in range(df.shape[0]):
        df1 = df.iloc[i:i+n, :]

        if df1.alt.iloc[0] < 100:
            continue
        elif df1.roc.mean() <= 0:
            continue
        elif df1.trk.std() > 5:
            continue
        elif df1.alt.iloc[0] > 5000:
            i = None
            break
        break
    return i


df0 = pd.read_csv(fin)

df0.dropna(subset=['lat', 'gs'], inplace=True)
df0['t'] = df0['t'].round()
df0 = df0.groupby(['t'], as_index=False).mean()


istart = search_start(df0, ndata)
if istart is None:
    print('Trajectory is not suitable for mass esitmation.')
    sys.exit(1)


df = df0.iloc[istart:istart+20, :]

t = df.t.values

lat = df.lat.values
lon = df.lon.values
alt = df.alt.values
gs = df.gs.values
trk = df.trk.values
roc = df.roc.values
vwx = df.vwx.values
vwy = df.vwy.values
# tau = df.temp.values
tau = aero.temperature(df.alt*aero.ft)

bearings = aero.bearing(lat[0], lon[0], lat, lon)
distances = aero.distance(lat[0], lon[0], lat, lon, alt*aero.ft)

x = distances * np.sin(np.radians(bearings))
y = distances * np.cos(np.radians(bearings))

z = alt * aero.ft
vg = gs * aero.kts
vz = roc * aero.fpm
chi = np.radians(trk)

vgx = vg * np.sin(chi)
vgy = vg * np.cos(chi)

times = t - t[0]
obs = np.array([x, y, z, vgx, vgy, vz, vwx, vwy, tau])
fn_log = '%s_%s_%s.log' % (ac, stdn, fn)

sir = smc.SIR(ac=ac, eng=eng, time=times, obs=obs, noise=stdn, logfn=fn_log)

smc_vis.trajectory(sir)

sir.init_particles(n_particles=500000)
sir.run(processdt=1)

smc_vis.convergence_all(sir)
smc_vis.particle_hist(sir)
smc_vis.mass_per_thrust(sir)
# smc_vis.convergence_m_eta(sir)
