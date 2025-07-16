# %%
from finesse.ligo.factory import aligo
import finesse.analysis.actions as fa
from finesse.ligo.maps import get_test_mass_surface_profile_interpolated
from finesse.ligo.actions import InitialLockLIGO
import numpy as np
from finesse.knm import Map
import finesse
from finesse.ligo import git_path
from mpi4py import MPI
from finesse.solutions import SimpleSolution
from scipy.optimize import minimize
import matplotlib.pyplot as plt
ifo = "LLO"

# %%
factory = aligo.ALIGOFactory(git_path() / ifo / "YAML" / f"{ifo.lower()}_O4.yaml")
factory.options.INPUT.add_IMC_and_IM1 = False
model = factory.make()

RH_I_uD_W = 1e-6 # uD/W ITM
RH_E_uD_W = 0.84e-6 # uD/W ETM
RH_f_uD_W = -9.9e-6 # uD/W ITM
P_IX_RH = 0.7
P_IY_RH = 0.7
P_EX_RH = 0
P_EY_RH = 0

model.ITMX.Rc = 2 / (2 / abs(model.ITMX.Rc) + RH_I_uD_W * P_IX_RH)
model.ETMX.Rc = 2 / (2 / abs(model.ETMX.Rc) + RH_E_uD_W * P_EX_RH)
model.ITMY.Rc = 2 / (2 / abs(model.ITMY.Rc) + RH_I_uD_W * P_IY_RH)
model.ETMY.Rc = 2 / (2 / abs(model.ETMY.Rc) + RH_E_uD_W * P_EY_RH)

model.ITMXlens.f = 1/(1/model.ITMXlens.f + RH_f_uD_W * P_IX_RH)
model.ITMYlens.f = 1/(1/model.ITMYlens.f + RH_f_uD_W * P_IY_RH)

model.L0.P = 2
R = 0.17
N = 301

x, y = (
    np.linspace(-R, R, N),
    np.linspace(-R, R, N),
)
mask = finesse.utilities.maps.circular_aperture(x, y, R)

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
model.modes("even", maxtem=4)
model.parse("fd E_car_refl PRM.p2.o f=0")
model.parse("fd E_u9_refl PRM.p2.o f=+f1")
model.parse("fd E_l9_refl PRM.p2.o f=-f1")
model.parse("fd E_car_prc PRM.p1.o f=0")
model.parse("fd E_u9_prc PRM.p1.o f=+f1")
model.parse("fd E_l9_prc PRM.p1.o f=-f1")
eigx = model.run("eigenmodes(cavXARM, 0)")
eigy = model.run("eigenmodes(cavYARM, 0)")

model.X_arm_loss = 70e-6
model.Y_arm_loss = 70e-6

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

lock = InitialLockLIGO(
    exception_on_lock_fail=True,
    lock_steps=100,
    gain_scale=0.4,
    pseudo_lock_arms=False,
    run_locks=True,
    exception_on_check_fail=True,
)

actions = [
    lock,
    fa.Noxaxis(name="DC"),
]

sol = model.run(fa.Series(*actions))
model.gIM2_p1_i.priority = 1e10
model.parse("mathd PRG_full Pprc/Pin")
base = model.deepcopy()

# %% Optimiser
with model.temporary_parameters():
    print("Optimise input qx and qy")
    sol = model.run()
    model.modes("even", maxtem=8)
    print("Pin:", sol['Pin'])
    print("Before")
    print("Px:", sol['Px'], " PRG:", sol['PRG_full'], " Prefl:", sol['PreflPRM'])
    opt = model.run('maximize(Px, [gIM2_p1_i.w0x, gIM2_p1_i.zx, gIM2_p1_i.w0y, gIM2_p1_i.zy])')
    sol = model.run()
    print("After")
    print("Px:", sol['Px'], " PRG:", sol['PRG_full'], " Prefl:", sol['PreflPRM'])
    print(
        "Mismatch from original X:",
        finesse.BeamParam.mismatch(
            model.gIM2_p1_i.qx,
            base.gIM2_p1_i.qx,
        ),
        "Y:",
        finesse.BeamParam.mismatch(
            model.gIM2_p1_i.qy,
            base.gIM2_p1_i.qy,
        )
    )
    
# %%
with model.temporary_parameters():
    model.gIM2_p1_i.w0x = opt.x[0]
    model.gIM2_p1_i.w0y = opt.x[2]
    model.gIM2_p1_i.zx = opt.x[1]
    model.gIM2_p1_i.zy = opt.x[3]
    sol = model.run("xaxis(gIM2_p1_i.zy, lin, -1, 1, 100, True)")
    plt.plot(sol.x[0], sol['PreflPRM'])
    
    
# %%
with model.temporary_parameters():
    model.modes("even", maxtem=6)
    homs = model.hom_labels
    model.gIM2_p1_i.w0x = opt.x[0]
    model.gIM2_p1_i.w0y = opt.x[2]
    model.gIM2_p1_i.zx = opt.x[1]
    model.gIM2_p1_i.zy = opt.x[3]
    DC = model.run()
    print(
        DC['PRG'],
        DC['PreflPRM'],
    )
    P = abs(DC['E_car_refl'])**2
    P /= np.linalg.norm(P)

    plt.bar(homs, P)
    plt.grid()
    plt.yscale("log")