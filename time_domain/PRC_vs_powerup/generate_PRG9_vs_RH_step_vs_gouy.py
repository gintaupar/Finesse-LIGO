# %%
import finesse
import numpy as np
from pathlib import Path
import finesse.analysis.actions as fa
from gpstime import gpsnow
import dill

from finesse.analysis.actions import TimeQuasiStatic

from prc_models import time_model

# %%
start_time = int(gpsnow())

ARM_POWER = 400e3  # [W]
LIMITING_APERTURE = 0.13  # [m]
ABS_COEFF = 0.5e-6  # [fractional]
ROUGH_RH_ACT = 13e-6  # RH lensing [D/W]
ROUGH_SH_ACT = 200e-6  # self heating lensing [D/W]

# compute the lens that should cancel out the self heating, roughly
RH_MAX = (ARM_POWER * ABS_COEFF * ROUGH_SH_ACT) / ROUGH_RH_ACT


def job(P_RH, dL):
    dt1 = 200
    dt2 = 2000

    t = np.unique(
        np.hstack(
            (
                np.arange(0, 8000 + dt1, dt1, dtype=float),
                np.arange(8000, 100000 + dt2, dt2, dtype=float),
            ),
        )
    )
    model = time_model(LIMITING_APERTURE)

    model.spaces.PR2_p2__PR3_p1.L += dL
    cold_gouy = model.PRC.gouy.mean() / 2

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

    return model.run(action), cold_gouy


P_RHs = np.array([1])
dlengths = np.array([-2e-3, 0, 2e-3, 4e-3, 6e-3, 8e-3, 10e-3])
results = np.array(
    [[job(prh, gouy) for prh in P_RHs] for gouy in dlengths], dtype=object
)
solutions = np.array(results[:, :, 0])
gouys = np.array(results[:, :, 1])
sol_file = Path(f"solutions.{start_time}.pkl")
loc = "data" / Path(Path(__file__).name[:-3])
loc.mkdir(exist_ok=True, parents=True)

dill.dump(
    (
        solutions,
        P_RHs,
        dlengths,
        gouys,
        start_time,
        ARM_POWER,
        LIMITING_APERTURE,
        ABS_COEFF,
        ROUGH_RH_ACT,
        ROUGH_SH_ACT,
    ),
    open(loc / sol_file, "wb"),
)
