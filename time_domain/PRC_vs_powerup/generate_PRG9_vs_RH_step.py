# %%
import finesse
import numpy as np
import matplotlib.cm as cm
import finesse.analysis.actions as fa
from gpstime import gpsnow
import dill

from finesse.analysis.actions import TimeQuasiStatic

from prc_models import time_model

finesse.init_plotting(fmts=["png"], dpi=100)
start_time = int(gpsnow())

ARM_POWER = 400e3  # [W]
LIMITING_APERTURE = 0.13  # [m]
ABS_COEFF = 0.5e-6  # [fractional]
ROUGH_RH_ACT = 13e-6  # RH lensing [D/W]
ROUGH_SH_ACT = 200e-6  # self heating lensing [D/W]

# compute the lens that should cancel out the self heating, roughly
RH_MAX = (ARM_POWER * ABS_COEFF * ROUGH_SH_ACT) / ROUGH_RH_ACT


def job(P_RH):
    dt1 = 100
    dt2 = 1000

    t = np.unique(
        np.hstack(
            (
                np.arange(0, 8000 + dt1, dt1, dtype=float),
                np.arange(8000, 100000 + dt2, dt2, dtype=float),
            ),
        )
    )
    model = time_model(LIMITING_APERTURE)

    model.HEATER.P = ARM_POWER
    lock_action = fa.PseudoLockCavity(model.PRC, mode=[0, 0])
    action = TimeQuasiStatic(
        times=t,
        show_progress=True,
        lock_action=lock_action,
        time_variable=model.t,
        other_analysis=fa.Eigenmodes(model.PRC, 0),
        events={1000: fa.Change({"THERMAL.P_RH": P_RH})},
    )
    return model.run(action)


P_RHs = np.linspace(0, RH_MAX, 10)
solutions = [job(_) for _ in P_RHs]

dill.dump(
    (
        solutions,
        P_RHs,
        start_time,
        ARM_POWER,
        LIMITING_APERTURE,
        ABS_COEFF,
        ROUGH_RH_ACT,
        ROUGH_SH_ACT,
    ),
    open(f"data/solutions.{start_time}.pkl", "wb"),
)
