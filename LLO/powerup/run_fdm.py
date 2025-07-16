# %%
import finesse
import finesse.analysis.actions as fac
import gpstime
import pickle
import argparse
import importlib
import numpy as np

from finesse_ligo.factory import ALIGOFactory
from finesse_ligo.actions import InitialLockLIGO
from copy import deepcopy
from pathlib import Path


# add code to store a flag for the warnings for lock convergence failure
import warnings

# Define a flag variable
warning_triggered = False
warning_flag=[warning_triggered]

# Store the original warning handler
original_warn_handler = warnings.showwarning

# ramping code
def get_ramp_power(t, Pmin, Pmax, tmin, tmax):

    # get the ramp slope
    m0 = (Pmax - Pmin)/(tmax - tmin)

    if t <= tmin:
        return Pmin
    elif t > tmin and t<= tmax:
        POut = m0*(t-tmin) + Pmin
        return POut
    else:
        return Pmax
    

# Custom warning handler
def custom_warn_handler(message, category, filename, lineno, file=None, line=None):
    global warning_triggered
    # Check if the warning message contains "Locks failed to converge"
    if "Locks failed to converge" in str(message):
        warning_triggered = True

    # Call the original warning handler to ensure the warning is processed as usual
    original_warn_handler(message, category, filename, lineno, file, line)

# Replace the default warning handler with the custom one
warnings.showwarning = custom_warn_handler


def is_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interprete


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
    #default="fenicsx.FENICSXThermalState",  # <<<<<<<<<<< Set default thermal state to use here
    # default="dmd.DMDThermalState",
    default="fdm.FDMThermalState",
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
llo.parse("""
    mathd Parm (Px+Py)/2

    fd E_refl_c0  IFI.p4.o f=0
    fd E_refl_u9  IFI.p4.o f=+f1
    fd E_refl_l9  IFI.p4.o f=-f1
    fd E_refl_u45 IFI.p4.o f=+f2
    fd E_refl_l45 IFI.p4.o f=-f2
                        
    fd E_prc_c0  PRM.p1.o f=0
    fd E_prc_u9  PRM.p1.o f=+f1
    fd E_prc_l9  PRM.p1.o f=-f1
    fd E_prc_u45 PRM.p1.o f=+f2
    fd E_prc_l45 PRM.p1.o f=-f2
            
    fd E_src_c0  SRM.p1.o f=0
    fd E_src_u9  SRM.p1.o f=+f1
    fd E_src_l9  SRM.p1.o f=-f1
    fd E_src_u45 SRM.p1.o f=+f2
    fd E_src_l45 SRM.p1.o f=-f2
            
    fd E_x_c0  ETMX.p1.i f=0
    fd E_x_u9  ETMX.p1.i f=+f1
    fd E_x_l9  ETMX.p1.i f=-f1
    fd E_x_u45 ETMX.p1.i f=+f2
    fd E_x_l45 ETMX.p1.i f=-f2
            
    fd E_y_c0  ETMY.p1.i f=0
    fd E_y_u9  ETMY.p1.i f=+f1
    fd E_y_l9  ETMY.p1.i f=-f1
    fd E_y_u45 ETMY.p1.i f=+f2
    fd E_y_l45 ETMY.p1.i f=-f2
            
    fd E_inx_c0  ITMXlens.p1.i f=0
    fd E_inx_u9  ITMXlens.p1.i f=+f1
    fd E_inx_l9  ITMXlens.p1.i f=-f1
    fd E_inx_u45 ITMXlens.p1.i f=+f2
    fd E_inx_l45 ITMXlens.p1.i f=-f2
        
    fd E_iny_c0  ITMYlens.p1.i f=0
    fd E_iny_u9  ITMYlens.p1.i f=+f1
    fd E_iny_l9  ITMYlens.p1.i f=-f1
    fd E_iny_u45 ITMYlens.p1.i f=+f2
    fd E_iny_l45 ITMYlens.p1.i f=-f2

    fd E_c0_as OM1.p1.i f=0
""")

TS = thermal_state_class(factory, llo)
initial_model = llo.deepcopy()  # keep a copy of the initial state
TS.update_maps()

# %% compute the round trip losses with the maps in
# and make sure overall loss is reasonable
llo.modes("even", maxtem=6)
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
outs = [llo.run()]
TS.values.out = outs[0]
eigensols = []
locks = []
values_used = []

TS.reset()

RHX = [0]
RHY=[0]
FRX = [0]
FRY=[0]
CO2X = [0]
CO2Y = [0]
dtList = [20]
lensList = [0]

import time as compTime
startT = compTime.time()

opdData = np.array([]).reshape((0,2))

while TS.t <= 2e4:
    loopT = compTime.time()
    print('-----------')
    print(compTime.time() - startT)
    print(TS.t)
    time.append(TS.t)
    values_used.append(deepcopy(TS.values))

    tRamp0 = 820
    tRamp1 = 4000
    Plow = 25
    Phigh = 100
    # Using hardcoded P_in data for power up.
    if TS.t > 180 and llo.L0.P != Plow and llo.L0.P < Plow:
        llo.L0.P = Plow
        llo.DARM_rf_lock.gain *= 2 / Plow
        llo.CARM_lock.gain /= Plow / 2
        llo.PRCL_lock.gain /= Plow / 2
        llo.SRCL_lock.gain /= Plow / 2
        llo.MICH_lock.gain /= Plow / 2

    #elif TS.t > 180 + 10 * 64 and llo.L0.P != Phigh:
    elif TS.t > 820:
        Pnow = get_ramp_power(TS.t, Plow, Phigh, tRamp0, tRamp1)
        Pold = llo.L0.P.value
        llo.L0.P = Pnow
        llo.CARM_lock.gain /= Pnow / Pold
        llo.DARM_rf_lock.gain /= Pnow / Pold
        llo.PRCL_lock.gain /= Pnow / Pold
        llo.SRCL_lock.gain /= Pnow / Pold
        llo.MICH_lock.gain /= Pnow / Pold

        TS.ts_itmx.PRH = 0
        TS.ts_itmy.PRH = 0

    '''if TS.t > 1000:
        TS.ts_itmx.PFR = 0
        TS.ts_itmy.PFR = 0
        TS.ts_itmx.PCO2 = 2
        TS.ts_itmy.PCO2 = 2'''
    PCO2L = 0
    PCO2H = 2
    PCO2now = get_ramp_power(TS.t, PCO2L, PCO2H, tRamp0+180, tRamp1+180)
    TS.ts_itmx.PCO2 = PCO2now
    TS.ts_itmy.PCO2 = PCO2now

    dTenv = 0.05*np.cos(2*np.pi*TS.t/86400)
    TS.ts_itmx.dTenv = dTenv
    TS.ts_itmy.dTenv = dTenv

    RHX.append(TS.ts_itmx.PRH)
    FRX.append(TS.ts_itmx.PFR)
    CO2X.append(TS.ts_itmx.PCO2)
    RHY.append(TS.ts_itmy.PRH)
    FRY.append(TS.ts_itmy.PFR)
    CO2Y.append(TS.ts_itmy.PCO2)
    dtList.append(TS.dt)
    lensList.append(1/llo.ITMXlens.f.value)


    opdData = np.vstack((opdData, [TS.t, np.max(TS.get_opd('ETMX'))]))
    models.append(llo.deepcopy())

    sols = llo.run(
        """
        series(
            run_locks(exception_on_fail=False, max_iterations=500),
            noxaxis()
        )
        """
    )
    llo.run(fac.SetLockGains(gain_scale=0.4))
    locks.append(sols["run locks"])
    TS.values.out = sols["noxaxis"]
    outs.append(TS.values.out)
    warning_flag.append(warning_triggered)

    print(
        TS.t,
        sols["noxaxis"]["Parm"],
        sols["noxaxis"]["PRG"],
        float(llo.ITMXlens.f.value),
        float(llo.ITMYlens.f.value),
    )

    # update the time step and the thermal lenses
    if TS.t < 1500:
        TS.dt = 20
    elif TS.t < 4000:
        TS.dt = 50
    else:
        TS.dt = 100

    TS.step()
    TS.update_maps()

# %% Save data
path = Path("./figures/")
path.mkdir(exist_ok=True)
out = path / f"llo_power_up_{args.time}.pkl"
print("Writing output to", out)

with open(out, "wb") as file:
    pickle.dump(
        {
            "parameters": factory.params,
            "options": factory.options,
            "outs": outs,
            "t": time,
        },
        file,
    )

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

finesse.init_plotting()

def shade_unlocked_region(time, warning_flag):
    # Check if there are any True elements in warning_flag
    if np.any(warning_flag):
        plt.fill_between(time, y1=0, y2=1, where=warning_flag, color='red', alpha=0.3, 
                         transform=plt.gca().get_xaxis_transform(), label='Unlocked region')
        plt.legend()

with PdfPages(f"llo_power_up_{args.time}.pdf") as pdf:



    plt.plot(time, tuple(out["Pin"] for out in outs))
    plt.ylabel("Power [W]")
    plt.xlabel("Time [s]")
    plt.title("Input power")
    shade_unlocked_region(time, warning_flag)
    pdf.savefig()
    plt.close()

    plt.plot(time, RHX, label='Ring Heater-X')
    plt.plot(time, RHY, label='Ring Heater-Y')
    plt.plot(time, FRX, label='FROSTI-X')
    plt.plot(time, FRY, label='FROSTI-Y')
    plt.plot(time, CO2X, label='CO2-X')
    plt.plot(time, CO2Y, label='CO2-Y')
    plt.ylabel("Power [W]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.title("Actuators (FROSTI/RH)")
    shade_unlocked_region(time, warning_flag)
    pdf.savefig()
    plt.close()

    plt.plot(time, lensList, label='Defocus-ITMX')
    plt.ylabel("Defocus [D]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.title("Thermal lens (ITMX)")
    shade_unlocked_region(time, warning_flag)
    pdf.savefig()
    plt.close()


    plt.plot(time, tuple(out["Prefl"] / 1e-3 for out in outs))
    plt.ylabel("Power [mW]")
    plt.xlabel("Time [s]")
    plt.title("REFL")
    # Overlay red region where warning_flag is True
    #plt.fill_between(time, y1=0, y2=1, where=warning_flag, color='red', alpha=0.3, transform=plt.gca().get_xaxis_transform())
    shade_unlocked_region(time, warning_flag)
    pdf.savefig()
    plt.close()

    plt.plot(time, tuple(out["Px"] / 1e3 for out in outs), label="X arm")
    plt.plot(time, tuple(out["Py"] / 1e3 for out in outs), label="Y arm", ls="--")
    plt.ylabel("Power [kW]")
    plt.xlabel("Time [s]")
    plt.title("Arm power")
    plt.legend()
    shade_unlocked_region(time, warning_flag)
    pdf.savefig()
    plt.close()

    plt.plot(time, tuple(out["AGX"] for out in outs), label="X arm")
    plt.plot(time, tuple(out["AGY"] for out in outs), label="Y arm")
    plt.ylabel("Gain")
    plt.xlabel("Time [s]")
    plt.title("Arm gains")
    plt.legend()
    shade_unlocked_region(time, warning_flag)
    pdf.savefig()
    plt.close()

    plt.plot(time, tuple(out["PRG9"] for out in outs), label="9")
    plt.plot(time, tuple(out["PRG45"] for out in outs), label="45")
    plt.plot(time, tuple(out["PRG"] for out in outs), label="Carrier")
    plt.ylabel("Gain")
    plt.xlabel("Time [s]")
    plt.title("Recycling gains")
    plt.legend()
    shade_unlocked_region(time, warning_flag)
    pdf.savefig()
    plt.close()

    plt.plot(time, tuple(out["PRG"] for out in outs), label="Carrier")
    plt.ylabel("Gain")
    plt.xlabel("Time [s]")
    plt.title("Recycling gains")
    plt.legend()
    shade_unlocked_region(time, warning_flag)
    pdf.savefig()
    plt.close()

    plt.plot(
        time,
        tuple(
            np.sum(
                out["E_prc_u9"] * out["E_prc_l9"].conjugate()
                + out["E_prc_u9"].conjugate() * out["E_prc_l9"]
            ).real
            / out["Pin"]
            for out in outs
        ),
    )
    plt.ylabel("Power [arb.]")
    plt.xlabel("Time [s]")
    plt.title("RF18 / P_IN")
    shade_unlocked_region(time, warning_flag)
    pdf.savefig()
    plt.close()

    plt.plot(
        time,
        tuple(
            np.sum(
                out["E_prc_u45"] * out["E_prc_l45"].conjugate()
                + out["E_prc_u45"].conjugate() * out["E_prc_l45"]
            ).real
            / out["Pin"]
            for out in outs
        ),
    )
    plt.ylabel("Power [arb.]")
    plt.xlabel("Time [s]")
    plt.title("RF90 / P_IN")
    shade_unlocked_region(time, warning_flag)
    pdf.savefig()
    plt.close()

    plt.plot(time, dtList, label='Time step')
    plt.ylabel("dt [s]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.title("Time step")
    shade_unlocked_region(time, warning_flag)
    pdf.savefig()
    plt.close()

# %%
