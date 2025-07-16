# %%
import finesse.ligo
from finesse.ligo.factory import ALIGOFactory

import finesse.materials
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

ITM_sub = 300.39e-6
ITM_srf = -46.52e-6
ETM_sub = 209.57e-6
ETM_srf = -33.46e-6
IRH_sub = -9.92e-6
IRH_srf = 0.992e-6
ERH_sub = -12.53e-6
ERH_srf = 0.841e-6

# We first make a factory object that can generate an ALIGO model
# here we do so using the LHO O4 parameter file
factory = ALIGOFactory(finesse.ligo.git_path() / "LHO" / "yaml" / "lho_O4.yaml")
factory.update_parameters(finesse.ligo.git_path() / "LHO" / "yaml" / "lho_mcmc_RC_lengths.yaml")

# %%
factory.reset()
lho = factory.make()
base = lho.deepcopy()

# %%

OM3_ARM = []
MM_ARM = []
Ps = np.linspace(0, 0.4, 30)
with lho.temporary_parameters():
    for P in Ps:
        print(P)
        lho.ITMX.Rc = 2/(2/base.ITMX.Rc + P * ITM_srf)
        lho.ITMY.Rc = 2/(2/base.ITMY.Rc + P * ITM_srf)
        lho.ETMX.Rc = 2/(2/base.ETMX.Rc + P * ETM_srf * 3/5)
        lho.ETMY.Rc = 2/(2/base.ETMY.Rc + P * ETM_srf * 3/5)
        lho.ITMXlens.f = 1/(1/base.ITMXlens.f + P * ITM_sub)
        lho.ITMYlens.f = 1/(1/base.ITMYlens.f + P * ITM_sub)

        def opt(x):
            lho.OM2.Rc = x
            lho.beam_trace()
            mx, my = lho.cavity_mismatch('cavXARM', 'cavOMC')
            mmx = 1-np.sqrt((1-mx)*(1-my))
            mx, my = lho.cavity_mismatch('cavYARM', 'cavOMC')
            mmy = 1-np.sqrt((1-mx)*(1-my))
            return (mmx + mmy)/2
        
        opt = minimize(opt, x0=lho.OM2.Rc.mean(), method='nelder-mead')
        OM3_ARM.append(opt.x)
        MM_ARM.append(opt.fun)

OM3_SRC = []
MM_SRC = []
Ps = np.linspace(0, 0.4, 30)
with lho.temporary_parameters():
    for P in Ps:
        print(P)
        lho.ITMX.Rc = 2/(2/base.ITMX.Rc + P * ITM_srf)
        lho.ITMY.Rc = 2/(2/base.ITMY.Rc + P * ITM_srf)
        lho.ETMX.Rc = 2/(2/base.ETMX.Rc + P * ETM_srf * 3/5)
        lho.ETMY.Rc = 2/(2/base.ETMY.Rc + P * ETM_srf * 3/5)
        lho.ITMXlens.f = 1/(1/base.ITMXlens.f + P * ITM_sub)
        lho.ITMYlens.f = 1/(1/base.ITMYlens.f + P * ITM_sub)

        def opt(x):
            lho.OM2.Rc = x
            lho.beam_trace()
            mx, my = lho.cavity_mismatch('cavSRX', 'cavOMC')
            mmx = 1-np.sqrt((1-mx)*(1-my))
            mx, my = lho.cavity_mismatch('cavSRY', 'cavOMC')
            mmy = 1-np.sqrt((1-mx)*(1-my))
            return (mmx + mmy)/2
        
        opt = minimize(opt, x0=lho.OM2.Rc.mean(), method='nelder-mead')
        OM3_SRC.append(opt.x)
        MM_SRC.append(opt.fun)

# %%
finesse.init_plotting()
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7, 6))

axs[0].plot(Ps, OM3_ARM, label='ARM')
axs[0].plot(Ps, OM3_SRC, label='SRC')
axs[0].hlines(base.OM2.Rc.mean(), min(Ps), max(Ps), label='nominal', ls='--')

axs[0].set_ylabel("OM2 RoC [m]")
axs[1].plot(Ps, np.array(MM_ARM)*100, label='ARM')
axs[1].plot(Ps, np.array(MM_SRC)*100, label='SRC')
axs[1].set_ylabel("Mismatch [%]")
axs[1].set_xlabel("Power absorbed [W]")
axs[0].set_title("Optimal OM2 vs power")
axs[1].set_title("Optimised CAV to OMC mismatch")
axs[0].legend()
axs[1].legend()
plt.savefig("./figures/optimal_OM2.pdf")
# %%
