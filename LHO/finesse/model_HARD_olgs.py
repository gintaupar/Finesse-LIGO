'''
Code used to model the LHO hard loop open loop gains
Do not use, kept for reference. Use model_HARD_olgs_with_factory.py
'''
# %%
import finesse.ligo
import numpy as np
import importlib
import matplotlib.pyplot as plt
import finesse.components.electronics as fce
from finesse_ligo import lho
from finesse.plotting import bode
from finesse.ligo.suspension import QUADSuspension
from finesse.ligo.asc import add_arm_ASC_DOFs
from finesse.ligo.ASC_controllers import get_controller
from finesse.analysis.actions import (
    FrequencyResponse,
    Series,
)
import sys

finesse.init_plotting()
# %%
# create model from aligo
katfile = importlib.resources.read_text(
    "finesse_ligo.katscript", "aligo_reversed_itm.kat"
)
base = finesse.ligo.make_aligo(katscript=katfile)

model = base.deepcopy()
model.modes(maxtem=4)
model.fsig.f = 1
model.L0.P = 54
P_target = 360e3
sol = model.run("run_locks(exception_on_fail=False)")
out = model.run()
sol.plot_error_signals()
print("Power in: " + str(model.L0.P))
print("X arm power: " + str(round(out["Px"]) / 1e3) + " kW")
print("Y arm power: " + str(round(out["Py"]) / 1e3) + " kW")
print("PRG: " + str(round(out["PRG"], 1)))
# %%
# add AS and REFL path
model = lho.add_AS_WFS(model)
model = lho.add_REFL_path(model)

# add TMS QPDs
model = lho.add_transmon_path(model, arm="x")
model = lho.add_transmon_path(model, arm="y")

# radius of curvature change at high power
ITM_Rc_D_per_W = -46e-6
ETM_Rc_D_per_W = -33.46e-6

model.ITMXlens.f = model.ITMXlens.f
model.ITMYlens.f = model.ITMYlens.f
model.ITMX.Rc = 2 / (2 / model.ITMX.Rc + ITM_Rc_D_per_W * P_target * 0.5e-6)
model.ITMY.Rc = 2 / (2 / model.ITMY.Rc + ITM_Rc_D_per_W * P_target * 0.5e-6)
model.ETMX.Rc = 2 / (2 / model.ETMX.Rc + ETM_Rc_D_per_W * P_target * 0.5e-6 * 3 / 5)
model.ETMY.Rc = 2 / (2 / model.ETMY.Rc + ETM_Rc_D_per_W * P_target * 0.5e-6 * 3 / 5)

# ASC controllers
DHP_z, DHP_p, DHP_k = get_controller("dhard_p")
DHY_z, DHY_p, DHY_k = get_controller("dhard_y")
CHP_z, CHP_p, CHP_k = get_controller("chard_p")
CHY_z, CHY_p, CHY_k = get_controller("chard_y")

# added scaling factors
model.add(fce.ZPKFilter("DHARD_P_cntrl", DHP_z, DHP_p, 5.3 * DHP_k, gain=1))
model.add(fce.ZPKFilter("DHARD_Y_cntrl", DHY_z, DHY_p, 4.9 * DHY_k, gain=1))
model.add(fce.ZPKFilter("CHARD_P_cntrl", CHP_z, CHP_p, -2.4 * CHP_k, gain=1))
model.add(fce.ZPKFilter("CHARD_Y_cntrl", CHY_z, CHY_p, -16.1 * CHY_k, gain=1))

# %%
sus_component = QUADSuspension

ITMX_sus = model.add(sus_component("ITMX_sus", model.ITMX.mech))
ETMX_sus = model.add(sus_component("ETMX_sus", model.ETMX.mech))
ITMY_sus = model.add(sus_component("ITMY_sus", model.ITMY.mech))
ETMY_sus = model.add(sus_component("ETMY_sus", model.ETMY.mech))

(
    CHARD_P,
    CSOFT_P,
    DHARD_P,
    DSOFT_P,
    CHARD_Y,
    CSOFT_Y,
    DHARD_Y,
    DSOFT_Y,
) = add_arm_ASC_DOFs(model)

# optimize the demod phase
model = lho.optimize_AS_WFS(model)
model = lho.optimize_REFL_WFS(model)

# %%
# sanity check of free sus plants at target power
F_Hz = np.geomspace(0.1, 10, 200)

sol1 = model.run(
    FrequencyResponse(
        F_Hz,
        [DHARD_P.AC.i, CHARD_P.AC.i, DHARD_Y.AC.i, CHARD_Y.AC.i],
        [DHARD_P.AC.o, CHARD_P.AC.o, DHARD_Y.AC.o, CHARD_Y.AC.o],
    )
)

bode(sol1.f, sol1["DHARD_P.AC.i", "DHARD_P.AC.o"], label="DHARD_P", wrap=True)
bode(sol1.f, sol1["CHARD_P.AC.i", "CHARD_P.AC.o"], label="CHARD_P", wrap=True)
bode(sol1.f, sol1["DHARD_Y.AC.i", "DHARD_Y.AC.o"], label="DHARD_Y", wrap=True)
bode(sol1.f, sol1["CHARD_Y.AC.i", "CHARD_Y.AC.o"], label="CHARD_Y", wrap=True)

sol2 = model.run(
    FrequencyResponse(
        F_Hz,
        [DSOFT_P.AC.i, CSOFT_P.AC.i, DSOFT_Y.AC.i, CSOFT_Y.AC.i],
        [DSOFT_P.AC.o, CSOFT_P.AC.o, DSOFT_Y.AC.o, CSOFT_Y.AC.o],
    )
)

bode(sol2.f, sol2["DSOFT_P.AC.i", "DSOFT_P.AC.o"], label="DSOFT_P", wrap=True)
bode(sol2.f, sol2["CSOFT_P.AC.i", "CSOFT_P.AC.o"], label="CSOFT_P", wrap=True)
bode(sol2.f, sol2["DSOFT_Y.AC.i", "DSOFT_Y.AC.o"], label="DSOFT_Y", wrap=True)
bode(sol2.f, sol2["CSOFT_Y.AC.i", "CSOFT_Y.AC.o"], label="CSOFT_Y", wrap=True)

# # %%
# freq = np.geomspace(1, 10e3, 500)
# darm1 = model.run(FrequencyResponse(freq, model.DARM.AC, model.AS.DC))

# bode(freq, darm1["DARM.AC", "AS.DC"], db=False)
# %%
# create HARD loop error signals and connect
model.connect(model.AS_A_WFS45y.Q, model.DHARD_P_cntrl.p1, name="IN_DH_P")

model.connect(model.REFL_B_WFS9y.I, model.CHARD_P_cntrl.p1, name="IN_CH_P_9B")
model.connect(model.REFL_B_WFS45y.I, model.CHARD_P_cntrl.p1, name="IN_CH_P_45B")
# model.IN_CH_P_9B.gain = 1
# model.IN_CH_P_45B.gain = 1

model.connect(model.AS_A_WFS45x.Q, model.DHARD_Y_cntrl.p1, name="IN_DH_Y")

model.connect(model.REFL_B_WFS9x.I, model.CHARD_Y_cntrl.p1, name="IN_CH_Y_9B")
model.connect(model.REFL_B_WFS45x.I, model.CHARD_Y_cntrl.p1, name="IN_CH_Y_45B")
# model.IN_CH_Y_9B.gain = 1
# model.IN_CH_Y_45B.gain = 1

model.connect(model.DHARD_P_cntrl.p2, DHARD_P.AC.i)
model.connect(model.CHARD_P_cntrl.p2, CHARD_P.AC.i)
model.connect(model.DHARD_Y_cntrl.p2, DHARD_Y.AC.i)
model.connect(model.CHARD_Y_cntrl.p2, CHARD_Y.AC.i)

if "pytest" not in sys.modules:
    model.display_signal_blockdiagram()

# %%
F_Hz = np.geomspace(0.1, 10, 200)


sol = model.run(
    Series(
        FrequencyResponse(
            F_Hz,
            model.DHARD_P_cntrl.p1,
            DHARD_P.AC.o,
            open_loop=True,
            name="DHARD_P_olg",
        ),
        FrequencyResponse(
            F_Hz,
            model.CHARD_P_cntrl.p1,
            CHARD_P.AC.o,
            open_loop=True,
            name="CHARD_P_olg",
        ),
        FrequencyResponse(
            F_Hz,
            model.DHARD_Y_cntrl.p1,
            DHARD_Y.AC.o,
            open_loop=True,
            name="DHARD_Y_olg",
        ),
        FrequencyResponse(
            F_Hz,
            model.CHARD_Y_cntrl.p1,
            CHARD_Y.AC.o,
            open_loop=True,
            name="CHARD_Y_olg",
        ),
    )
)


# %%
if "pytest" not in sys.modules:
    colors = ("xkcd:burgundy", "xkcd:dark olive", "xkcd:salmon pink", "xkcd:cornflower")
    names = ("DHARD_P_olg", "CHARD_P_olg", "DHARD_Y_olg", "CHARD_Y_olg")
    dofs_in = ("DHARD_P.AC.i", "CHARD_P.AC.i", "DHARD_Y.AC.i", "CHARD_Y.AC.i")
    dofs_out = ("DHARD_P.AC.o", "CHARD_P.AC.o", "DHARD_Y.AC.o", "CHARD_Y.AC.o")
    for j in range(4):
        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].semilogx(
            F_Hz,
            20 * np.log10(np.abs(sol[names[j]][dofs_in[j], dofs_out[j]])),
            color=colors[j],
        )
        ax[1].semilogx(
            F_Hz,
            np.angle(sol[names[j]][dofs_in[j], dofs_out[j]], deg=True),
            color=colors[j],
        )
        ax[0].set_title(dofs_in[j][:7])
        ax[0].set_ylabel("Mag")
        ax[1].set_xlabel("Freq [Hz]")
        ax[1].set_ylabel("Phase [deg]")

# %%
# errorbar functions
def csd_variance_mag(coh, averages):
    """Relative CSD variance along the mag axis
    Variance CSD / Mean CSD^2
    coh in this case is gamma**2, the power coherence.
    Bendat and Piersol Random Data Analysis and Measurement Procedures Eq 9.31
    Copied from C. Cahillane
    """
    return 1 / (averages * coh)

def csd_variance_phase(coh, averages):
    """Relative CSD variance along the minor axis
    Variance CSD / Mean CSD^2
    Copied from C. Cahillane
    """
    return (1 - coh) / (2 * averages * coh)


dir = 'LHO/Measurements/'
DHP_file = np.loadtxt(dir+'DHARD_P_tf_20230810.txt')
freq_DHP = DHP_file[:,0]
in1_exc_DHP = DHP_file[:,1] + 1j*DHP_file[:,2]
in2_exc_DHP = DHP_file[:,3] + 1j*DHP_file[:,4]

DHP_olg = in1_exc_DHP / in2_exc_DHP

DHP_C = np.loadtxt(dir+'DHARD_P_coherence_20230810.txt')
in1_exc_C_DHP = DHP_C[:,1]
in2_exc_C_DHP = DHP_C[:,2]

power = np.round(out["Px"] / 1e3)

N_avg = 20
# dhard p errorbars
dhard_p_mag_std = np.sqrt(csd_variance_mag(in1_exc_C_DHP, N_avg) + csd_variance_mag(in2_exc_C_DHP, N_avg))
dhard_p_phase_std = np.sqrt(csd_variance_phase(in1_exc_C_DHP, N_avg) + csd_variance_phase(in2_exc_C_DHP, N_avg))

fig=plt.figure()
ax=fig.add_subplot(211)
ax.loglog(freq_DHP, np.abs(DHP_olg), '.', label='Unbiased OLG taken at 60W PSL', color='xkcd:cornflower')
ax.fill_between(freq_DHP, np.abs(DHP_olg)*(1 + dhard_p_mag_std), np.abs(DHP_olg)*(1 - dhard_p_mag_std), color='xkcd:cornflower', alpha=0.3, label='$\pm 1$ $\sigma$' )
ax.loglog(
            F_Hz,
            np.abs(sol[names[0]][dofs_in[0], dofs_out[0]]),
            color=colors[0], label=f'Modeled with Finesse3, {power} kW'
        )
ax.set_title('DHARD P OLG')
ax.legend()
ax.set_xlim(0.1, 10)
ax.set_ylim(2e-3, 1e2)
ax.set_yticks([1e-2, 1, 1e2])
ax.set_ylabel('Mag')
ax.grid(True, which='both')

ax=fig.add_subplot(212)
ax.semilogx(freq_DHP, np.angle(DHP_olg, deg=True), '.', color='xkcd:cornflower')
ax.fill_between(freq_DHP, (180/np.pi)*(np.angle(DHP_olg) + dhard_p_phase_std), (180/np.pi)*np.angle(DHP_olg) - dhard_p_phase_std, color='xkcd:cornflower', alpha=0.3)
ax.semilogx(
            F_Hz,
            np.angle(sol[names[0]][dofs_in[0], dofs_out[0]], deg=True),
            color=colors[0],
        )
ax.set_ylabel('Phase [deg]')
ax.set_xlabel('Freq [Hz]')
ax.set_ylim(-200, 200)
ax.set_xlim(0.1, 10)
ax.grid(True, which='both')

# %%

N_avg = 25

CHP_file = np.loadtxt(dir+'CHARD_P_tf_20230810.txt')
freq_CHP = CHP_file[:,0]
in1_exc_CHP = CHP_file[:,1] + 1j*CHP_file[:,2]
in2_exc_CHP = CHP_file[:,3] + 1j*CHP_file[:,4]

CHP_olg = in1_exc_CHP / in2_exc_CHP

CHP_C = np.loadtxt(dir+'CHARD_P_coherence_20230810.txt')
in1_exc_C_CHP = CHP_C[:,1]
in2_exc_C_CHP = CHP_C[:,2]

# chard p errobars
chard_p_mag_std = np.sqrt(csd_variance_mag(in1_exc_C_CHP, N_avg) + csd_variance_mag(in2_exc_C_CHP, N_avg))
chard_p_phase_std = np.sqrt(csd_variance_phase(in1_exc_C_CHP, N_avg) + csd_variance_phase(in2_exc_C_CHP, N_avg))

fig=plt.figure()
ax=fig.add_subplot(211)
ax.loglog(freq_CHP, np.abs(CHP_olg), '.', label='Unbiased OLG taken at 60W PSL', color='xkcd:teal')
ax.fill_between(freq_CHP, np.abs(CHP_olg)*(1 + chard_p_mag_std), np.abs(CHP_olg)*(1 - chard_p_mag_std), color='xkcd:teal', alpha=0.3, label='$\pm 1$ $\sigma$' )
ax.loglog(
            F_Hz,
            np.abs(sol[names[1]][dofs_in[1], dofs_out[1]]),
            color=colors[1], label=f'Modeled with Finesse3, {power} kW'
        )
ax.set_title('CHARD P OLG')
ax.legend()
ax.set_xlim(0.1, 10)
ax.set_ylim(2e-3, 1e2)
ax.set_yticks([1e-2, 1, 1e2])
ax.set_ylabel('Mag')
ax.grid(True, which='both')

ax=fig.add_subplot(212)
ax.semilogx(freq_CHP, np.angle(CHP_olg, deg=True), '.', color='xkcd:teal')
ax.fill_between(freq_CHP, (180/np.pi)*(np.angle(CHP_olg) + chard_p_phase_std), (180/np.pi)*np.angle(CHP_olg) - chard_p_phase_std, color='xkcd:teal', alpha=0.3)
ax.semilogx(
            F_Hz,
            np.angle(sol[names[1]][dofs_in[1], dofs_out[1]], deg=True),
            color=colors[1],
        )
ax.set_ylabel('Phase [deg]')
ax.set_xlabel('Freq [Hz]')
ax.set_ylim(-200, 200)
ax.set_xlim(0.1, 10)
ax.grid(True, which='both')
# %%
