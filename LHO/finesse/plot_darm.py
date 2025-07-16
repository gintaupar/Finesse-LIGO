# %%
import finesse.ligo
from finesse.ligo.lho import make_O4_lho
from finesse.analysis.actions import FrequencyResponse
import numpy as np
from tools import InitialLockLIGO
import baryrat
from finesse.plotting import bode

finesse.init_plotting()

# %%
lho = make_O4_lho(add_quads=False)
lho.modes("even", maxtem=4)
lho.L0.P = 60
lho.fsig.f = None
# Need to make sure we're at a good operating point
# so we lock the LHO model
lho.run(InitialLockLIGO);

# %%
darm_solution = lho.run(
    FrequencyResponse(
        np.geomspace(10, 10e3, 200),
        lho.DARM.AC,
        lho.AS.DC,
        name='fr'
    )
)
# %%
print("DARM drive")
print(lho.DARM.drives[0].AC_IN.full_name, lho.DARM.drives[1].AC_IN.full_name)
print(lho.DARM.amplitudes)

# %% Fit some zpks
fit = baryrat.aaa(darm_solution.f, darm_solution['DARM.AC', 'AS.DC'], mmax=2)
cav_pole = fit.poles().imag[0]

# %%
f = darm_solution.f
H = darm_solution['DARM.AC', 'AS.DC']
axs = bode(f, H, db=False)
axs = bode(f, fit(f), db=False, axs=axs, ls=':', label='Single pole fit')
axs[1].set_xlabel("Frequency [Hz]")
axs[0].set_ylabel("Amplitude [W/m]")
axs[0].set_title(f"Sensing function, cavity pole {cav_pole:.0f} Hz")
# %%
