# %%
import finesse
import finesse.analysis.actions as fac
import gpstime
import pickle
import argparse
import importlib
import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from finesse_ligo.factory import ALIGOFactory
from finesse_ligo.actions import InitialLockLIGO
from copy import deepcopy
from pathlib import Path
from utils import is_notebook

# Setup a command line interface and defaults to use from a notebook if running
# in vscode, etc.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--time",
    default=int(gpstime.gpsnow()),
    help="GPS time to use as a reference",
)
parser.add_argument(
    "--model",
    default="fenicsx.FENICSXThermalState",  # <<<<<<<<<<< Set default thermal state to use here
    # default="dmd.DMDThermalState",
    help="thermal model class to use. Should be input as module.classname",
)

if is_notebook():
    # Use defaults if in a notebook
    args = parser.parse_args("")
else:
    args = parser.parse_args()

# Try and get the info and import the thermal state model to use
module_name, class_name = args.model.split(".")
module = importlib.import_module(module_name)
thermal_state_class = getattr(module, class_name)

# %% Load in the parameter file
factory = ALIGOFactory(finesse.ligo.git_path() / "LLO" / "yaml" / "llo_O4.yaml")
factory.params.INPUT.LASER.power = 2
# Make the ifo model
factory.options.LSC.add_output_detectors = True
factory.options.ASC.add = False
factory.options.thermal.add = True

llo = factory.make()
# Fields to study in time
llo.parse(
    """
    fd E_refl_u45 M5.p2.o f=+f2
    fd E_refl_l45 M5.p2.o f=-f2
    fd E_refl_u9 M5.p2.o f=+f1
    fd E_refl_l9 M5.p2.o f=-f1
    fd E_refl_c0 M5.p2.o f=0
    
    fd E_prc_u45 PRM.p1.o f=+f2
    fd E_prc_l45 PRM.p1.o f=-f2
    fd E_prc_u9 PRM.p1.o f=+f1
    fd E_prc_l9 PRM.p1.o f=-f1
    fd E_prc_c0 PRM.p1.o f=0
    
    fd E_xarm_u45 ETMX.p1.o f=+f2
    fd E_xarm_l45 ETMX.p1.o f=-f2
    fd E_xarm_u9 ETMX.p1.o f=+f1
    fd E_xarm_l9 ETMX.p1.o f=-f1
    fd E_xarm_c0 ETMX.p1.o f=0
    
    fd E_yarm_u45 ETMY.p1.o f=+f2
    fd E_yarm_l45 ETMY.p1.o f=-f2
    fd E_yarm_u9 ETMY.p1.o f=+f1
    fd E_yarm_l9 ETMY.p1.o f=-f1
    fd E_yarm_c0 ETMY.p1.o f=0
    
    var P_ACO2X 0
    var P_ACO2Y 1
    mathd Parm (Px+Py)/2
    readout_dc TRX ETMX.p2.o
    readout_dc TRY ETMY.p2.o
"""
)

TS = thermal_state_class(factory, llo, N=81)
initial_model = llo.deepcopy()  # keep a copy of the initial state
TS.update_maps()

# %% compute the round trip losses with the maps in
# and make sure overall loss is reasonable
llo.modes("even", maxtem=8)
eigx = llo.run("eigenmodes(cavXARM, 0)")
eigy = llo.run("eigenmodes(cavYARM, 0)")

loss_x = llo.X_arm_loss + eigx.loss(True)[1][0]
loss_y = llo.Y_arm_loss + eigy.loss(True)[1][0]
print("X arm loss: ", loss_x / 1e-6, "ppm")
print("Y arm loss: ", loss_y / 1e-6, "ppm")
# Apply corrections to get back to original losses
print("Old X arm plane-wave loss: ", llo.X_arm_loss / 1e-6, "ppm")
print("Old Y arm plane-wave loss: ", llo.Y_arm_loss / 1e-6, "ppm")
llo.X_arm_loss -= eigx.loss(True)[1][0]
llo.Y_arm_loss -= eigy.loss(True)[1][0]
print("New X arm plane-wave loss: ", llo.X_arm_loss / 1e-6, "ppm")
print("New Y arm plane-wave loss: ", llo.Y_arm_loss / 1e-6, "ppm")

# %%
lock = InitialLockLIGO(
    exception_on_lock_fail=False,
    lock_steps=100,
    gain_scale=0.4,
    pseudo_lock_arms=False,
    run_locks=True,
)
sol = llo.run(lock)

# save initial gains to be changed during power up
initial_gains = {}
for lsc_dof in ["DARM_rf", "CARM", "PRCL", "SRCL", "MICH"]:
    initial_gains[lsc_dof] = llo.get(f"{lsc_dof}_lock.gain")

# %%
initial_P = 2
llo.L0.P = initial_P
time = [0]
models = [llo.deepcopy()]
sol = llo.run('series(noxaxis(), dc_fields(name="dc"))')
outs = [sol["noxaxis"]]
TS.values.out = outs[0]
eigensols = []
locks = []
values_used = []
DCs = [sol["dc"]]
TS.reset()
TS.dt = 25

# %%
while TS.t <= 10000:
    print(TS.t)
    time.append(TS.t)
    values_used.append(deepcopy(TS.values))

    # # Using hardcoded P_in data for power up.
    # if TS.t > 180 and llo.L0.P != 25 and llo.L0.P < 25:
    #     llo.L0.P = 25
    #     llo.DARM_rf_lock.gain *= 2 / 25
    #     llo.CARM_lock.gain /= 25 / 2
    #     llo.PRCL_lock.gain /= 25 / 2
    #     llo.SRCL_lock.gain /= 25 / 2
    #     llo.MICH_lock.gain /= 25 / 2

    # elif TS.t > 180 + 10 * 64 and llo.L0.P != 64:
    #     llo.L0.P = 64
    #     llo.CARM_lock.gain /= 64 / 25
    #     llo.DARM_rf_lock.gain /= 64 / 25
    #     llo.PRCL_lock.gain /= 64 / 25
    #     llo.SRCL_lock.gain /= 64 / 25
    #     llo.MICH_lock.gain /= 64 / 25

    # if TS.t < 2000:
    #     TS.dt = 100
    # el
    if TS.t < 1000:
        TS.dt = 25
    elif TS.t < 2000:
        TS.dt = 50
    elif TS.t < 4000:
        TS.dt = 100
    elif TS.t < 6000:
        TS.dt = 100
    else:
        TS.dt = 100
        llo.P_ACO2X = 2
        llo.P_ACO2Y = 2

    TS.step(evaluate_deformation=False)
    TS.update_maps()

    models.append(llo.deepcopy())
    models[-1].beam_trace()

    sols = llo.run(
        """
        series(
            run_locks(exception_on_fail=False, max_iterations=500),
            noxaxis(),
            dc_fields(name='dc'),
        )
        """
    )
    llo.run(fac.SetLockGains(gain_scale=0.4))
    locks.append(sols["run locks"])
    TS.values.out = sols["noxaxis"]
    outs.append(TS.values.out)
    DCs.append(sols["dc"])

    print(
        TS.t,
        sols["noxaxis"]["Parm"],
        sols["noxaxis"]["PRG"],
        float(llo.ITMXlens.f.value),
        float(llo.ITMYlens.f.value),
    )

    if TS.t < 1500:
        TS.dt = 20
    elif TS.t < 3000:
        TS.dt = 100
    else:
        TS.dt = 1000

    TS.step()
    TS.update_maps()

# %% Save data
path = Path("./figures/")
path.mkdir(exist_ok=True)
out = path / f"llo_power_up_{args.time}.pkl"
print("Writing output to", out)

model_parameters = [
    {
        p.full_name: p.value.eval() if hasattr(p.value, "eval") else p.value
        for p in model.all_parameters
    }
    for model in models
]

model_node_q = []

for model in models:
    model.beam_trace()
    model_node_q.append({node.full_name: node.q for node in models[0].optical_nodes})

with open(out, "wb") as file:
    pickle.dump(
        {
            "parameters": factory.params,
            "options": factory.options,
            "outs": outs,
            "t": time,
            "DCs": DCs,
            "model_parameters": model_parameters,
            "homs": llo.homs,
            "model_node_q": model_node_q,
        },
        file,
    )

# %%
finesse.init_plotting()
N = len(outs)

DC = DCs[0]
c0 = DC.frequencies == 0
u9 = DC.frequencies == llo.f1
l9 = DC.frequencies == -llo.f1
u45 = DC.frequencies == llo.f2
l45 = DC.frequencies == -llo.f2


def beat(f1, f2):
    return np.array(
        tuple(
            np.sum(
                DC["PRM.p1.o", f1] * DC["PRM.p1.o", f2].conjugate()
                + DC["PRM.p1.o", f1].conjugate() * DC["PRM.p1.o", f2]
            ).real
            / out["Pin"]
            for DC, out in zip(DCs, outs)
        )
    )


RF18 = beat(u9, l9)
RF90 = beat(u45, l45)
plot_name = path / f"llo_power_up_{args.time}.pdf"
with PdfPages(plot_name) as pdf:
    plt.plot(time[:N], tuple(out["Pin"] for out in outs))
    plt.ylabel("Power [W]")
    plt.xlabel("Time [s]")
    plt.title("Input power")
    pdf.savefig()
    plt.close()

    plt.plot(time[:N], tuple(out["Prefl"] / 1e-3 for out in outs))
    plt.ylabel("Power [mW]")
    plt.xlabel("Time [s]")
    plt.title("REFL")
    pdf.savefig()
    plt.close()

    plt.plot(time[:N], tuple(out["Px"] / 1e3 for out in outs), label="X arm")
    plt.plot(
        time[:N],
        tuple(out["Py"] / 1e3 for out in outs),
        label="Y arm",
        ls="--",
    )
    plt.ylabel("Power [kW]")
    plt.xlabel("Time [s]")
    plt.title("Arm power")
    plt.legend()
    pdf.savefig()
    plt.close()

    plt.plot(time[:N], tuple(out["AGX"] for out in outs), label="X arm")
    plt.plot(time[:N], tuple(out["AGY"] for out in outs), label="Y arm")
    plt.ylabel("Gain")
    plt.xlabel("Time [s]")
    plt.title("Arm gains")
    plt.legend()
    pdf.savefig()
    plt.close()

    plt.plot(time[:N], tuple(out["PRG9"] for out in outs), label="9")
    plt.plot(time[:N], tuple(out["PRG45"] for out in outs), label="45")
    plt.plot(time[:N], tuple(out["PRG"] for out in outs), label="Carrier")
    plt.ylabel("Gain")
    plt.xlabel("Time [s]")
    plt.title("Recycling gains")
    plt.legend()
    pdf.savefig()
    plt.close()

    plt.plot(time[:N], tuple(out["PRG"] for out in outs), label="Carrier")
    plt.ylabel("Gain")
    plt.xlabel("Time [s]")
    plt.title("Recycling gains")
    plt.legend()
    pdf.savefig()
    plt.close()

    plt.plot(
        time[:N],
        RF18[:N],
    )
    plt.ylabel("Power [arb.]")
    plt.xlabel("Time [s]")
    plt.title("RF18 / P_IN")
    pdf.savefig()
    plt.close()

    plt.plot(
        time[:N],
        RF90[:N],
    )
    plt.ylabel("Power [arb.]")
    plt.xlabel("Time [s]")
    plt.title("RF90 / P_IN")
    pdf.savefig()
    plt.close()

    plt.plot(time[:N], [out["REFL9_Q"] for out in outs[:N]], label="REFL9_Q")
    plt.plot(time[:N], [out["REFL9_I"] for out in outs[:N]], label="REFL9_I")
    plt.plot(time[:N], [out["REFL45_I"] for out in outs[:N]], label="REFL45_Q")
    plt.plot(time[:N], [out["REFL45_Q"] for out in outs[:N]], label="REFL45_I")
    plt.ylabel("Power [arb.]")
    plt.xlabel("Time [s]")
    plt.title("REFL signals")
    plt.legend()
    pdf.savefig()
    plt.close()
