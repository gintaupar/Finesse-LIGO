# %%
# Computes a "power up" but in a quasi-static way. This is not technically correct
# as the actuation strength varies with spot sizes.
# Runs with some ring heater on to see how that changes
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
lho = make_lho_quadratic_thermal_apertured()

# %%
lock = InitialLockLIGO(exception_on_lock_fail=False, lock_steps=100, gain_scale=0.5, pseudo_lock_arms=False)
#lock.actions = lock.actions[:3]
sol = lho.run(lock)
#lho.run("run_locks(max_iterations=1000)")

# %%
print_DC_state(lho)

# %%
Parms = np.linspace(10e3, 700e3, 10)

lho.ITM_sub = 300.39e-6
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
    lho.PRHX = 0
    lho.PRHY = 0
    sols_0W = run()
    lho.PRHX = 1
    lho.PRHY = 1
    sols_1W = run()
    lho.PRHX = 2
    lho.PRHY = 2
    sols_2W = run()
    lho.PRHX = 3
    lho.PRHY = 3
    sols_3W = run()
    lho.PRHX = 5
    lho.PRHY = 5
    sols_5W = run()

# %%
for sols in [sols_0W, sols_1W, sols_2W, sols_3W, sols_5W]:
    N = len(sols)
    sol = SolutionSet(sols)

    Pin = Parms[:N] / sol['PRG'].solutions / sol['AGX'].solutions * 2
    plt.plot(Pin, Parms[:N]/1e3, lw=2, ls='--')

plt.xlim(0, 140)
plt.ylim(0, 1000)
plt.scatter(57, 375, 80, color='m', marker='o')
plt.scatter(74, 430, 80, color='m', marker='o') # https://alog.ligo-wa.caltech.edu/aLOG/index.php?callRep=68531
plt.scatter(80, 445, 80, color='m', marker='o')
plt.legend(["0W", "1W", "2W", "3W", "5W", "LHO"])
plt.xlabel("Power input [W]")
plt.ylabel("Power arm [kW]")
plt.title("LHO - Build up vs common ITM RH")
plt.savefig("./figures/power_up_RH/Parm.pdf")

# %%
for sols in [sols_0W, sols_1W, sols_2W]:
    N = len(sols)
    sol = SolutionSet(sols)

    Pin = Parms[:N] / sol['PRG'].solutions / sol['AGX'].solutions * 2
    plt.plot(Pin, sol['Prefl'].solutions, lw=2, ls='--')

plt.xlim(0, 100)
plt.ylim(0, 1.2)

plt.legend(["0W", "1W", "2W"])
plt.xlabel("Power input [W]")
plt.ylabel("Reflected power [W]")
plt.title("LHO - REFL vs common ITM RH")
plt.savefig("./figures/power_up_RH/Prefl.pdf")
# %%