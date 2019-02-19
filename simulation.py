import numpy as np
import smc, smc_vis
import libsim

n_run = 15
times = np.arange(n_run)

ac = 'B737'
eng = 'CFM56-7B26'

print('Aircraft model:', ac)
print('Engine:', eng)


# sim_condition = [60000, 0.96, 'n2', 'n1']
sim_condition = [60000, 0.96, 'n2', 'n2']
# sim_condition = [60000, 0.96, 'n2', 'n3']
# sim_condition = [60000, 0.96, 'n2', 'n4']

m0, eta0, stdn_sim, stdn_sir = sim_condition

emin = libsim.calc_min_eta(m0, ac)
if eta0 < emin:
    raise RuntimeError('eta is small than minimum required eta: %.5f < %.5f' % (eta0, emin))

X0 = libsim.initX(m0, eta0, 80, 45, 0, 0)

fn_log = 'sim_%s_%d_%d_%s_%s.log' % (ac, m0/1000, eta0*100, stdn_sim, stdn_sir)

X_true, Y = libsim.sim_trjectory(n_run, ac, eng, X0, smc.stdns[stdn_sim])

sir = smc.SIR(ac=ac, eng=eng, time=times, obs=Y, noise=stdn_sir, logfn=fn_log)
sir.X_true = X_true

smc_vis.trajectory(sir)

sir.init_particles(n_particles=1000000)
sir.run(processdt=1)

smc_vis.convergence_all(sir)
smc_vis.mass_per_thrust(sir)
# smc_vis.convergence_m_eta(sir)
