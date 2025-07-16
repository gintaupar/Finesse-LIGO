# %%
import finesse.ligo
from finesse.ligo.factory import ALIGOFactory

import finesse.materials
import numpy as np
import matplotlib.pyplot as plt

finesse.init_plotting()

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
factory.options.INPUT.add_IMC_and_IM1 = True
lho = factory.make()
base = lho.deepcopy()

# %%
with lho.temporary_parameters():
    mmx = []
    mmy = []
    Ps = np.linspace(0, 0.4, 50)
    for P in Ps:

        lho.ITMX.Rc = 2/(2/base.ITMX.Rc + P * ITM_srf)
        lho.ITMY.Rc = 2/(2/base.ITMY.Rc + P * ITM_srf)
        lho.ETMX.Rc = 2/(2/base.ETMX.Rc + P * ETM_srf * 3/5)
        lho.ETMY.Rc = 2/(2/base.ETMY.Rc + P * ETM_srf * 3/5)
        lho.ITMXlens.f = 1/(1/base.ITMXlens.f + P * ITM_sub)
        lho.ITMYlens.f = 1/(1/base.ITMYlens.f + P * ITM_sub)
        mmx.append(lho.cavity_mismatch()[0])
        mmy.append(lho.cavity_mismatch()[1])

# %%
def get_MM(A, B, mmx, mmy):
    MM = np.array([
        [_[A, B] for _ in mmx],
        [_[A, B] for _ in mmy],
    ])
    MM = 1 - np.sqrt(np.product(1-MM, 0))
    return MM

# %%
fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharey=False, sharex=True)
i = -1
axs = axs.T.flat
for N, O in [('S', 'O'), ('P', 'I')]:
    MMX_IY  = get_MM(f'cav{O}MC', 'cavXARM', mmx, mmy)
    MMX_IX  = get_MM(f'cav{O}MC', 'cavYARM', mmx, mmy)
    MMX_IPX = get_MM(f'cav{O}MC', f'cav{N}RX', mmx, mmy)
    MMX_IPY = get_MM(f'cav{O}MC', f'cav{N}RY', mmx, mmy)

    plt.sca(axs[i:=i+1])
    plt.plot(Ps/0.5e-6/1e3, MMX_IY *100, label='XARM')
    plt.plot(Ps/0.5e-6/1e3, MMX_IX *100, label='YARM')
    plt.plot(Ps/0.5e-6/1e3, MMX_IPX*100, label=f'{N}RX')
    plt.plot(Ps/0.5e-6/1e3, MMX_IPY*100, label=f'{N}RY')
    plt.ylabel("Mismatch [%]")
    plt.title(f"{O}MC to {N}RC & ARM mismatch")
    plt.legend()
    plt.ylim(0, 18)

    MMX_PXX = get_MM(f'cav{N}RX', 'cavXARM', mmx, mmy)
    MMX_PXY = get_MM(f'cav{N}RX', 'cavYARM', mmx, mmy)
    MMX_PYX = get_MM(f'cav{N}RY', 'cavXARM', mmx, mmy)
    MMX_PYY = get_MM(f'cav{N}RY', 'cavYARM', mmx, mmy)

    plt.sca(axs[i:=i+1])
    plt.plot(Ps/0.5e-6/1e3, MMX_PXX * 100, label=f'{N}RX-XARM')
    plt.plot(Ps/0.5e-6/1e3, MMX_PXY * 100, label=f'{N}RX-YARM')
    plt.plot(Ps/0.5e-6/1e3, MMX_PYX * 100, label=f'{N}RY-XARM')
    plt.plot(Ps/0.5e-6/1e3, MMX_PYY * 100, label=f'{N}RY-YARM')
    plt.xlabel(r"Arm power ($\alpha=0.5ppm$)[kW]")
    plt.ylabel("Mismatch [%]")
    plt.title(f"{N}RC to ARM mismatches")
    plt.legend()
    plt.ylim(0, 6)


plt.tight_layout()
plt.savefig("./figures/IMC_OMC_modematching_vs_power.pdf")
# %%
