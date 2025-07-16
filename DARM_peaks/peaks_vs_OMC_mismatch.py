# %%
import sys
import matplotlib.pyplot as plt
from finesse.ligo.factory import aligo
import finesse.analysis.actions as fa
from finesse.ligo.maps import get_test_mass_surface_profile_interpolated
from finesse.ligo.actions import InitialLockLIGO, DARM_RF_to_DC
import numpy as np
from finesse.knm import Map
import finesse
from finesse.ligo import git_path

finesse.init_plotting()
# %%
# Calculate the DC powers
use_DC_readout = True  # if True, use DC readout

factory = aligo.ALIGOFactory(git_path() / "LHO" / "yaml" / "lho_O4.yaml")
factory.update_parameters(git_path() / "LHO" / "yaml" / "lho_mcmc_RC_lengths.yaml")

model = factory.make()

R = 0.17
N = 301

x, y = (
    np.linspace(-R, R, N),
    np.linspace(-R, R, N),
)
mask = finesse.utilities.maps.circular_aperture(x, y, R)

# Get surfaces
for TM in [model.ITMX, model.ITMY, model.ETMX, model.ETMY]:
    if "X" in TM.name:
        P = factory.params.X
    else:
        P = factory.params.Y
    TM._unfreeze()
    TM.static = get_test_mass_surface_profile_interpolated(
        P[TM.name[:-1]].ID, make_axisymmetric=True
    )(x, y)
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

loss_x = model.X_arm_loss + eigx.loss(True)[1][0]
loss_y = model.Y_arm_loss + eigy.loss(True)[1][0]
print("X arm loss: ", loss_x / 1e-6, "ppm")
print("Y arm loss: ", loss_y / 1e-6, "ppm")
# Apply corrections to get back to original losses
print("Old X arm plane-wave loss: ", model.X_arm_loss / 1e-6, "ppm")
print("Old Y arm plane-wave loss: ", model.Y_arm_loss / 1e-6, "ppm")
model.X_arm_loss -= eigx.loss(True)[1][0]
model.Y_arm_loss -= eigy.loss(True)[1][0]
print("New X arm plane-wave loss: ", model.X_arm_loss / 1e-6, "ppm")
print("New Y arm plane-wave loss: ", model.Y_arm_loss / 1e-6, "ppm")

# %%
lock = InitialLockLIGO(
    exception_on_lock_fail=False,
    lock_steps=100,
    gain_scale=0.4,
    pseudo_lock_arms=False,
    run_locks=True,
)

actions = [
    lock,
    fa.Noxaxis(name="DC"),
]
if use_DC_readout:
    actions.insert(1, aligo.DARM_RF_to_DC())

sol = model.run(fa.Series(*actions))

# %%
with model.temporary_parameters():
    model.DARM_dc_lock.offset = 0.1
    print(model.mismatches_table())
    sol0 = model.run("frequency_response(linspace(10k, 11k, 300), L0.frq, AS.DC)")

    for model.OM1.Rc in [4.2, 4, 3.9, 3.5, 3, 2.5]:
        print(model.OM1.Rc)
        sol = model.run("run_locks(max_iterations=1000, exception_on_fail=False)")

    print(model.mismatches_table())
    sol1 = model.run("frequency_response(linspace(10k, 11k, 300), L0.frq, AS.DC)")

# %%
plt.semilogy(sol0.f, abs(sol0["AS.DC", "L0.frq"]), label="1% mismatch")
plt.semilogy(sol1.f, abs(sol1["AS.DC", "L0.frq"]), label="5% mismatch")
plt.xlabel("Frequency [Hz]")
plt.ylabel("FREQ->DCPD [W/Hz]")
plt.title("10kHz peaks vs output mismatch")
plt.legend()


# %%
TFS = fa.FrequencyResponse(np.geomspace(1, 30e3, 1000), ['L0.frq', 'DARM.AC'], 'AS.DC')
sol = model.run(TFS)
# %%
plt.loglog(sol.f, abs(sol["AS.DC", "L0.frq"])/ abs(sol["AS.DC", "DARM.AC"]))
