# Quick example of finding DC power
# %%
from finesse.ligo.factory import aligo
import finesse.analysis.actions as fa
from finesse.ligo.maps import get_test_mass_surface_profile_interpolated
from finesse.ligo.actions import InitialLockLIGO, DARM_RF_to_DC
import numpy as np
from finesse.knm import Map
import finesse

# %%
def ppath_join(ifo, *subpath):
    """Find the path to the yaml files
    """
    assert ifo in ["LHO", "LLO"]
    import git
    from os import path

    repo = git.Repo(".", search_parent_directories=True)
    root_dir = repo.git.rev_parse("--show-toplevel")
    return path.join(root_dir, ifo, "yaml", *subpath)

# %%
# Calculate the DC powers
use_DC_readout = True  # if True, use DC readout

import sys

# Get the first argument
ifo = sys.argv[1]

if ifo not in ["LHO", "LLO"]:
    raise ValueError("The first argument must be either LHO or LLO")

factory = aligo.ALIGOFactory(ppath_join(ifo, f"{ifo.lower()}_O4.yaml"))
if ifo == "LHO":
    factory.update_parameters(
        ppath_join("LHO", "lho_mcmc_RC_lengths.yaml")
    )
    
model = factory.make()

R = 0.17
N = 301

x, y = (
    np.linspace(-R, R, N),
    np.linspace(-R, R, N),
)
mask = finesse.utilities.maps.circular_aperture(x,y,R)

# Get surfaces
for TM in [model.ITMX, model.ITMY, model.ETMX, model.ETMY]:
    if 'X' in TM.name:
        P = factory.params.X
    else:
        P = factory.params.Y
    TM._unfreeze()
    TM.static = get_test_mass_surface_profile_interpolated(P[TM.name[:-1]].ID, make_axisymmetric=True)(x, y)
    TM._freeze()

model.ITMX.surface_map = Map(x, y, amplitude=mask, opd=model.ITMX.static)
model.ETMX.surface_map = Map(x, y, amplitude=mask, opd=model.ETMX.static)
model.ITMY.surface_map = Map(x, y, amplitude=mask, opd=model.ITMY.static)
model.ETMY.surface_map = Map(x, y, amplitude=mask, opd=model.ETMY.static)

model.ITMXlens.OPD_map = Map(x, y, amplitude=mask)
model.ITMYlens.OPD_map = Map(x, y, amplitude=mask)

x_r3, XR3_baffle = finesse.ligo.maps.aligo_O4_PR3_SR3_baffle(N=N)

model.PR3.surface_map = Map(x_r3, x_r3, amplitude=XR3_baffle)
model.SR3.surface_map = Map(x_r3, x_r3, amplitude=XR3_baffle)

# compute the round trip losses with the maps in and make sure overall loss is reasonable
model.modes("even", maxtem=8)
eigx = model.run("eigenmodes(cavXARM, 0)")
eigy = model.run("eigenmodes(cavYARM, 0)")

loss_x = (model.X_arm_loss + eigx.loss(True)[1][0])
loss_y = (model.Y_arm_loss + eigy.loss(True)[1][0])
print("X arm loss: ", loss_x/1e-6, "ppm")
print("Y arm loss: ", loss_y/1e-6, "ppm")
# Apply corrections to get back to original losses
print("Old X arm plane-wave loss: ", model.X_arm_loss/1e-6, "ppm")
print("Old Y arm plane-wave loss: ", model.Y_arm_loss/1e-6, "ppm")
model.X_arm_loss -= eigx.loss(True)[1][0]
model.Y_arm_loss -= eigy.loss(True)[1][0]
print("New X arm plane-wave loss: ", model.X_arm_loss/1e-6, "ppm")
print("New Y arm plane-wave loss: ", model.Y_arm_loss/1e-6, "ppm")

 # %%
lock = InitialLockLIGO(
    exception_on_lock_fail=False,
    lock_steps=100,
    gain_scale=0.4,
    pseudo_lock_arms=False,
    run_locks=True
)

actions = [
    lock,
    fa.Noxaxis(name="DC"),
]
if use_DC_readout:
    actions.insert(1, aligo.DARM_RF_to_DC())
    
sol = model.run(fa.Series(*actions))
# %%
# print the DC power (or values) of every detector
for detector in sol["DC"].outputs:
    print(detector, sol["DC", detector])
# %%
