# %%
# Computes a "power up" but in a quasi-static way. This is not technically correct
# as the actuation strength varies with spot sizes.
import finesse
import finesse.ligo
from finesse.ligo.actions import InitialLockLIGO
import finesse.analysis.actions as fac
from finesse.solutions.base import SolutionSet
import finesse.materials
import numpy as np
import matplotlib.pyplot as plt

from common import make_lho_quadratic_thermal_apertured, print_DC_state
finesse.init_plotting()

# %%
lho = make_lho_quadratic_thermal_apertured(apertured=False)

# %%
lock = InitialLockLIGO(exception_on_lock_fail=False, lock_steps=100, gain_scale=0.5, pseudo_lock_arms=False)
#lock.actions = lock.actions[:3]
sol = lho.run(lock)
#lho.run("run_locks(max_iterations=1000)")

# %%
print_DC_state(lho)

# %%
Parms = np.linspace(10e3, 1000e3, 15)

lho.PRHX = 0.0
lho.PRHY = 0.0

lho.alpha_ITMX = 0.5e-6
lho.alpha_ITMY = 0.5e-6
lho.alpha_ETMX = 0.3e-6
lho.alpha_ETMY = 0.3e-6

def run():
    sols = []
    with lho.temporary_parameters():
        for _P in Parms:
            print(_P)
            lho.PX = _P
            lho.PY = _P
            # lho.run(lock)
            lho.run("run_locks(max_iterations=200, exception_on_fail=False)")
            sols.append(lho.run(fac.Noxaxis()))
    return sols

with lho.temporary_parameters():
    lho.ITM_sub = 300.39e-6
    sols_full = run()
    lho.ITM_sub = 300.39e-6/2
    sols_half = run()   
    lho.ITM_sub = 0
    sols_zero = run()

# %%
for sols in [sols_full, sols_half, sols_zero]:
    N = len(sols)
    sol = SolutionSet(sols)

    Pin = Parms[:N] / sol['PRG'].solutions / sol['AGX'].solutions * 2
    plt.plot(Pin, Parms[:N]/1e3, lw=2, ls='--')

plt.xlim(0, 140)
plt.ylim(0, 1000)
plt.scatter(57, 375, 80, color='m', marker='o')
plt.scatter(74, 430, 80, color='m', marker='o') # https://alog.ligo-wa.caltech.edu/aLOG/index.php?callRep=68531
plt.scatter(80, 445, 80, color='m', marker='o')
plt.legend(["300 [uD/W]", "150 [uD/W]", "0 [uD/W]", "LHO"])
plt.xlabel("Power input [W]")
plt.ylabel("Power arm [kW]")
plt.title("LHO - Build up vs scaled ITM substrate lensing\nIdealised quadratic deformations, No TCS or apertures")
plt.savefig("./figures/power_up_no_aperture/Parm.pdf")

# %%
for sols in [sols_full, sols_half, sols_zero]:
    N = len(sols)
    sol = SolutionSet(sols)

    Pin = Parms[:N] / sol['PRG'].solutions / sol['AGX'].solutions * 2
    plt.plot(Pin, sol['PRG'].solutions, lw=2, ls='--')

plt.xlim(0, 140)
plt.ylim(40, None)
plt.legend(["Full", "Half", "Zero", "LHO 57W"])
plt.xlabel("Power input [W]")
plt.ylabel("Power Recycling Gain")
plt.title("LHO - Build up vs ITM lensing\nPurely quadratic deformations, No TCS or apertures")
plt.savefig("./figures/power_up_no_aperture/PRG.pdf")

# %%
for sols in [sols_full, sols_half, sols_zero]:
    N = len(sols)
    sol = SolutionSet(sols)

    Pin = Parms[:N] / sol['PRG'].solutions / sol['AGX'].solutions * 2
    plt.plot(Pin, sol['AGX'].solutions, lw=2, ls='--')

plt.xlim(0, 140)
plt.ylim(40, None)
plt.legend(["Full", "Half", "Zero", "LHO 57W"])
plt.xlabel("Power input [W]")
plt.ylabel("Arm gain")
plt.title("LHO - Build up vs ITM lensing\nPurely quadratic deformations, No TCS or apertures")
plt.savefig("./figures/power_up_no_aperture/ARMG.pdf")

# %%
for sols in [sols_full, sols_half, sols_zero]:
    N = len(sols)
    sol = SolutionSet(sols)

    Pin = Parms[:N] / sol['PRG'].solutions / sol['AGX'].solutions * 2
    plt.plot(Pin, sol['PRG9'].solutions, lw=2, ls='--')

plt.xlim(0, 140)
plt.ylim(70, None)
plt.legend(["Full", "Half", "Zero", "LHO 57W"])
plt.xlabel("Power input [W]")
plt.ylabel("Power Recycling Gain 9MHz")
plt.title("LHO - Build up vs ITM lensing\nPurely quadratic deformations, No TCS or apertures")
plt.savefig("./figures/power_up_no_aperture/PRG9.pdf")

# %%