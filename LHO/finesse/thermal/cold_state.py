# %% Computes the cold state lens PRG
import finesse
import finesse.ligo
from finesse.ligo.factory import ALIGOFactory
from finesse.ligo.actions import InitialLockLIGO, DARM_RF_to_DC
import finesse.analysis.actions as fac
from finesse.knm import Map

import finesse.materials
import numpy as np
from tabulate import tabulate

from finesse.ligo.maps import get_test_mass_surface_profile_interpolated, aligo_O4_TM_aperture, aligo_O4_BS_to_ITMX_baffle, aligo_O4_BS_to_ITMY_baffle, aligo_O4_ESD_inner_aperture
import matplotlib.pyplot as plt


finesse.init_plotting()

# %% We first make a factory object that can generate an ALIGO model
# here we do so using the LHO O4 parameter file
factory = ALIGOFactory(finesse.ligo.git_path() / "LHO" / "yaml" / "lho_O4.yaml")
factory.update_parameters(finesse.ligo.git_path() / "LHO" / "yaml" / "lho_mcmc_RC_lengths.yaml")

# %%
factory.reset()
lho = factory.make()

lho.parse("fd E_c0_as OM1.p1.i f=0")
lho.parse("mathd Parm Px+Py")

R = 0.16
N = 201

x, TM_aperture = aligo_O4_TM_aperture(R, N)
x, X_aperture = aligo_O4_ESD_inner_aperture(R, N)
x, Y_aperture = aligo_O4_ESD_inner_aperture(R, N)
y = x

# Get surfaces
ITMX_static = get_test_mass_surface_profile_interpolated(factory.params.X.ITM.ID, make_axisymmetric=True)(x, y)
ETMX_static = get_test_mass_surface_profile_interpolated(factory.params.X.ETM.ID, make_axisymmetric=True)(x, y)
ITMY_static = get_test_mass_surface_profile_interpolated(factory.params.Y.ITM.ID, make_axisymmetric=True)(x, y)
ETMY_static = get_test_mass_surface_profile_interpolated(factory.params.Y.ETM.ID, make_axisymmetric=True)(x, y)

# For test masses to always recompute, bit of a hack at the moment in FINESSE
lho.ITMX.misaligned.is_tunable = True
lho.ETMX.misaligned.is_tunable = True
lho.ITMY.misaligned.is_tunable = True
lho.ETMY.misaligned.is_tunable = True

lho.ITMX.surface_map = Map(x, y, amplitude=TM_aperture, opd=ITMX_static)
lho.ITMY.surface_map = Map(x, y, amplitude=TM_aperture, opd=ITMY_static)
lho.ETMX.surface_map = Map(x, y, amplitude=TM_aperture, opd=ETMX_static)
lho.ETMY.surface_map = Map(x, y, amplitude=TM_aperture, opd=ETMY_static)

lho.ITMXlens.OPD_map = Map(x, y, amplitude=X_aperture)
lho.ITMYlens.OPD_map = Map(x, y, amplitude=Y_aperture)

# compute the round trip losses with the maps in and make sure overall loss
# is reasonable
lho.L0.P = 2
lho.modes(maxtem=8)
eigx = lho.run("eigenmodes(cavXARM, 0)")
eigy = lho.run("eigenmodes(cavYARM, 0)")

loss_x = (lho.X_arm_loss + eigx.loss(True)[1][0])
loss_y = (lho.Y_arm_loss + eigy.loss(True)[1][0])
print("X arm loss: ", loss_x/1e-6, "ppm")
print("Y arm loss: ", loss_y/1e-6, "ppm")
# Apply corrections to get back to original losses
print("Old X arm plane-wave loss: ", lho.X_arm_loss/1e-6, "ppm")
print("Old Y arm plane-wave loss: ", lho.Y_arm_loss/1e-6, "ppm")
lho.X_arm_loss -= eigx.loss(True)[1][0]
lho.Y_arm_loss -= eigy.loss(True)[1][0]
print("New X arm plane-wave loss: ", lho.X_arm_loss/1e-6, "ppm")
print("New Y arm plane-wave loss: ", lho.Y_arm_loss/1e-6, "ppm")

# %%
lock = InitialLockLIGO(exception_on_lock_fail=False, lock_steps=100)
#lock.actions = lock.actions[:3]
sol = lho.run(lock)
#lho.run("run_locks(max_iterations=1000)")

# %%
#lho.run("minimize(Pas, [MICH2.DC, DARM.DC])")
DC = lho.run()
#DC = sol['after ARMs+AS']
data = [
    ("P_x", DC['Px']/1e3, 'kW'),
    ("P_y", DC['Py']/1e3, 'kW'),
    ("PRG", DC['PRG']),
    ("PRG9", DC['PRG9']),
    ("PRG45", DC['PRG45']),
    ("X arm gain", DC['AGX']),
    ("Y arm gain", DC['AGY']),
    ("P_IN", DC['Pin'], 'W'),
    ("P_REFL", DC['Prefl'], 'W'),
    ("P_REFL", DC['Prefl'], 'W'),
    ("P_PRC", DC['Pprc'], 'W'),
    ("P_DCPD", DC['Pas']/1e-3, 'mW')
]

print(tabulate(data, headers=["Name", "Value", "Unit"]))

# %%
results = {}
with lho.temporary_parameters():
    def sweep_cold_lens():
        DCs = []
        Ds = np.linspace(-5e-6, 50e-6, 20)
        for D in Ds:
            print(D)
            lho.ITMXlens.f = 1/D
            lho.ITMYlens.f = 1/D
            out = lho.run("series(run_locks(), noxaxis())")
            DCs.append(out['noxaxis'])
        return Ds, DCs

    results["with ESD & TM"] = sweep_cold_lens()

    lho.ITMXlens.OPD_map = None
    lho.ITMYlens.OPD_map = None
    lho.run("run_locks()")
    results["With TM, no ESD"] = sweep_cold_lens()

    lho.ITMX.surface_map = None
    lho.ETMX.surface_map = None
    lho.ITMY.surface_map = None
    lho.ETMY.surface_map = None
    lho.ITMXlens.OPD_map = None
    lho.ITMYlens.OPD_map = None
    lho.run("run_locks()")
    results["No apertures"] = sweep_cold_lens()

# %%
for name, (Ds, DCs) in results.items():
    plt.plot(Ds/1e-6, [_['PRG'] for _ in DCs], label=name)

plt.ylim(30, 60)
plt.vlines(1/lho.ITMXlens.f, *plt.gca().get_ylim(), ls='--', label='X')
plt.vlines(1/lho.ITMYlens.f, *plt.gca().get_ylim(), ls=':', color='r', label='Y')

plt.xlabel("Cold lens [uD]")
plt.ylabel("PRG")
plt.title("LHO cold lens vs apertures")
plt.legend()

# %%
