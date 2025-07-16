# %%

import finesse.ligo
from finesse.ligo.factory import aligo
import matplotlib.pyplot as plt
import finesse
import numpy as np
from finesse.components.mechanical import Pendulum

finesse.init_plotting(fmts=['png'], dpi=200)
# %%
factory = aligo.ALIGOFactory("lho_O4.yaml")
factory.options.ASC.add = True
# Switch on and off RP with this
# factory.options.QUAD_suspension_model = Pendulum
# factory.options.QUAD_suspension_kwargs = {"mass": 40}
model = factory.make()
model.modes(maxtem=4)  # use at least up to 2nd order modes to include mismatch
model.run(aligo.InitialLock())
model.run(aligo.DARM_RF_to_DC())
model.fsig.f = 1

# %%
# Change these to see some coupling differences
model.SR2.ybeta = 0e-9
model.SRM.ybeta = 100e-9

for model.DHARD_P.DC in np.linspace(-3e-9, 3e-9, 7):
    with model.temporary_parameters():
        print(model.DHARD_P.DC)
        model.ITMX.ybeta = model.ITMX.ybeta.eval()
        model.ITMY.ybeta = model.ITMY.ybeta.eval()
        model.ETMX.ybeta = model.ETMX.ybeta.eval()
        model.ETMY.ybeta = model.ETMY.ybeta.eval()
        DC = model.run()
        print(DC["Px"])
        model.run("run_locks()")
        sol = model.run(
            "frequency_response(geomspace(10, 5000, 11), SRCL.AC.i, AS.DC.o)"
        )
        plt.loglog(
            sol.f, abs(sol.out.squeeze()), label=f"{model.DHARD_P.DC/1e-9:.0f} nRad"
        )
plt.legend()
plt.xlabel("Frequency [Hz]")
plt.ylabel("Ampltiude [DARM/SRCL]")
plt.title(f"SRCL->DARM vs DHARD_P offset\nSRM_P={model.SRM.ybeta/1e-9:.0f} nRad SR2_P={model.SR2.ybeta/1e-9:.0f} nRad")

# %%
