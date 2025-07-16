# %%
import finesse
import finesse.ligo
from finesse.ligo.factory import ALIGOFactory
from finesse.ligo.actions import InitialLockLIGO, DARM_RF_to_DC
import finesse.analysis.actions as fac
from finesse.knm import Map
from finesse.cymath.homs import HGModes
import numpy as np
import finesse.materials
import numpy as np
from tabulate import tabulate
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from ifo_thermal_state.aligo_3D import (
    make_test_mass_model,
    AdvancedLIGOTestMass3D,
    AdvancedLIGOTestMass3DTime,
    AdvancedLIGOTestMass3DSteadyState
)
from ifo_thermal_state.mesh import sweep_cylinder_3D
from types import SimpleNamespace
from finesse.ligo.maps import get_test_mass_surface_profile_interpolated
import matplotlib.pyplot as plt
import sys
from finesse.ligo.maps import get_test_mass_surface_profile_interpolated, aligo_O4_TM_aperture, aligo_O4_BS_to_ITMX_baffle, aligo_O4_BS_to_ITMY_baffle, aligo_O4_ESD_inner_aperture

sys.path.append("..")
from tools import get_mask, get_deformation, get_opd

import ACO2

finesse.init_plotting()

# %% We first make a factory object that can generate an ALIGO model
# here we do so using the LHO O4 parameter file
factory = ALIGOFactory(finesse.ligo.git_path() / "LHO" / "yaml" / "lho_O4.yaml")
factory.update_parameters(finesse.ligo.git_path() / "LHO" / "yaml" / "lho_mcmc_RC_lengths.yaml")

# %%
# model = make_test_mass_model(
#     mesh_function_kwargs={
#         "HR_mesh_size": 0.02,
#         "AR_mesh_size": 0.02,
#         "mesh_algorithm": 6,
#     }
# )

z_RH = 30e-3
w_RH = 20e-3
H = 0.2
heights = [
    (z_RH - w_RH / 2) / H,
    (z_RH + (w_RH / 2)) / H,
    1
]
model = make_test_mass_model(
    mesh_function=sweep_cylinder_3D,
    mesh_function_kwargs={
        "num_elements": [2, 4, 15],
        "heights": heights,
        "add_flats": False,
        "HR_mesh_size": 0.02,
        "AR_mesh_size": 0.02,
        "mesh_algorithm": 6,
    },
)

CP = ACO2.make_CP_model()
#ts_cp = AdvancedLIGOTestMass3DTime(CP, 30)
# Make the fea models
ts_itmx = AdvancedLIGOTestMass3DTime(model, 30)
ts_etmx = AdvancedLIGOTestMass3DTime(model, 30)
ts_itmy = AdvancedLIGOTestMass3DTime(model, 30)
ts_etmy = AdvancedLIGOTestMass3DTime(model, 30)

def initial_condition(x):
    return np.full((x.shape[1],), 0)

ts_itmx.set_initial_condition(initial_condition)
ts_etmx.set_initial_condition(initial_condition)
ts_itmy.set_initial_condition(initial_condition)
ts_etmy.set_initial_condition(initial_condition)

# %%
factory.reset()
base_lho = factory.make()
base_lho.L0.P = 2
base_lho.parse("""
fd E_itmx1 ITMX.p1.i f=0
fd E_itmx2 ITMX.p1.o f=0
fd E_itmx3 ITMX.p2.i f=0
fd E_itmx4 ITMX.p2.o f=0
fd E_etmx  ETMX.p1.i f=0
          
fd E_itmy1 ITMY.p1.i f=0
fd E_itmy2 ITMY.p1.o f=0
fd E_itmy3 ITMY.p2.i f=0
fd E_itmy4 ITMY.p2.o f=0
fd E_etmy  ETMY.p1.i f=0
          
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

# Make equal for now
base_lho.ITMXlens.f = np.inf
base_lho.ITMYlens.f = np.inf

base = base_lho.deepcopy()

R = 0.17
N = 201

x, TM_aperture = aligo_O4_TM_aperture(R, N)
x, X_aperture = aligo_O4_ESD_inner_aperture(R, N)
x, Y_aperture = aligo_O4_ESD_inner_aperture(R, N)
y = x
X, Y = np.meshgrid(x, y)

TM_MASK = get_mask(x, y, ts_itmx)

# Get surfaces
ITMX_static = get_test_mass_surface_profile_interpolated(factory.params.X.ITM.ID, make_axisymmetric=True)(x, y)
ETMX_static = get_test_mass_surface_profile_interpolated(factory.params.X.ETM.ID, make_axisymmetric=True)(x, y)
ITMY_static = get_test_mass_surface_profile_interpolated(factory.params.Y.ITM.ID, make_axisymmetric=True)(x, y)
ETMY_static = get_test_mass_surface_profile_interpolated(factory.params.Y.ETM.ID, make_axisymmetric=True)(x, y)

# For test masses to always recompute, bit of a hack at the moment in FINESSE
base_lho.ITMX.misaligned.is_tunable = True
base_lho.ETMX.misaligned.is_tunable = True
base_lho.ITMY.misaligned.is_tunable = True
base_lho.ETMY.misaligned.is_tunable = True

base_lho.ITMX.surface_map = Map(x, y, amplitude=TM_aperture) #, opd=ITMX_static)
base_lho.ETMX.surface_map = Map(x, y, amplitude=TM_aperture) #, opd=ETMX_static)
base_lho.ITMY.surface_map = Map(x, y, amplitude=TM_aperture) #, opd=ITMY_static)
base_lho.ETMY.surface_map = Map(x, y, amplitude=TM_aperture) #, opd=ETMY_static)

base_lho.ITMXlens.OPD_map = Map(x, y, amplitude=X_aperture)
base_lho.ITMYlens.OPD_map = Map(x, y, amplitude=Y_aperture)

# compute the round trip losses with the maps in and make sure overall loss
# is reasonable
base_lho.modes("even", maxtem=8)
eigx = base_lho.run("eigenmodes(cavXARM, 0)")
eigy = base_lho.run("eigenmodes(cavYARM, 0)")

loss_x = (base_lho.X_arm_loss + eigx.loss(True)[1][0])
loss_y = (base_lho.Y_arm_loss + eigy.loss(True)[1][0])
print("X arm loss: ", loss_x/1e-6, "ppm")
print("Y arm loss: ", loss_y/1e-6, "ppm")
# Apply corrections to get back to original losses
print("Old X arm plane-wave loss: ", base_lho.X_arm_loss/1e-6, "ppm")
print("Old Y arm plane-wave loss: ", base_lho.Y_arm_loss/1e-6, "ppm")
base_lho.X_arm_loss -= eigx.loss(True)[1][0]
base_lho.Y_arm_loss -= eigy.loss(True)[1][0]
print("New X arm plane-wave loss: ", base_lho.X_arm_loss/1e-6, "ppm")
print("New Y arm plane-wave loss: ", base_lho.Y_arm_loss/1e-6, "ppm")

# %%
lho = base_lho.deepcopy()
lock = InitialLockLIGO(exception_on_lock_fail=False, lock_steps=100, gain_scale=0.4, pseudo_lock_arms=False)
sol = lho.run(lock)

# %%
def DC_print(DC):
    data = [
        ("P_x", DC['Px']/1e3, 'kW'),
        ("P_y", DC['Py']/1e3, 'kW'),
        ("PRG", DC['PRG']),
        ("PRG9", DC['PRG9']),
        ("PRG45", DC['PRG45']),
        ("X arm gain", DC['AGX']),
        ("Y arm gain", DC['AGY']),
        ("P_REFL", DC['Prefl'], 'W'),
        ("P_REFL", DC['Prefl'], 'W'),
        ("P_PRC", DC['Pprc'], 'W'),
        ("P_DCPD", DC['Pas']/1e-3, 'mW')
    ]

    print(tabulate(data, headers=["Name", "Value", "Unit"]))
    
def DC(model):
    DC_print(model.run())

DC(lho)

# %%
ss_itm = AdvancedLIGOTestMass3DSteadyState(model)
ss_etm = AdvancedLIGOTestMass3DSteadyState(model)

# %%
# Loading in ring heater realistic angular distribution
I_RH_theta = np.loadtxt("./RH_intensity.dat", delimiter=",")
theta_RH, _idx = np.unique(I_RH_theta[:, 0], return_index=True)
I_RH_data = I_RH_theta[:, 1][_idx]
#plt.plot(theta_RH, I_RH_data)
I_RH_theta_interp = interp1d(
    theta_RH, I_RH_data, kind="linear", fill_value="extrapolate"
)

# I_ACO2_data = np.loadtxt("./CO2_Annular_Projection_I_1W.txt", delimiter=" ")
# _x = I_ACO2_data[:512,0]
# _y = I_ACO2_data[::512,1]
# _data = I_ACO2_data[:,2].reshape((512, 512))

# I_ACO2_interp = RegularGridInterpolator(
#     (_x, _y), _data
# )

# ACO2_ravg = np.load("ACO2_ravg.npz")
# R = np.sqrt(X**2 + Y**2)
# I_ACO2_interp = RegularGridInterpolator(
#     (x, y),
#     np.interp(R, ACO2_ravg['r'], ACO2_ravg['OPD'].mean(0))
# )

# %%
def I_ACO2(x):
    return I_ACO2_interp((x[0], x[1])) * values.P_ACO2

def I_RH(x):
    A_RH = 2 * np.pi * model.radius * w_RH
    I = (
        np.ones_like(
            I_RH_theta_interp(np.arctan2(x[1], x[0]) + np.pi)
        ) * np.double(
            (x[2] >= -model.thickness / 2 + z_RH - w_RH / 2)
            & (x[2] <= -model.thickness / 2 + z_RH + w_RH / 2)
        )
        / A_RH
    )
    return I

def I_ZERO(x):
    return np.zeros_like(x[0])

def I_ITM_RH(x):
    return I_RH(x) * values.P_itm_RH

def I_ETM_RH(x):
    return I_RH(x) * values.P_etm_RH

def get_intensity(x, q, E):
    HGs = HGModes(q, lho.homs)
    a = HGs.compute_points(x[0], x[1]) * E
    _E = np.sum(a, axis=0)
    return (_E * _E.conj()).real

# Get intensity on HR surface:
def I_ITMX_HR(x):
    if values.out is not None:
        IFO = get_intensity(x, lho.ITMX.p1.i.q, values.out["E_itmx1"][:, None]) * 0.5e-6
    else:
        IFO = np.zeros_like(x[0])

    CHETA = get_intensity(
        x,
        finesse.BeamParam(w=values.w_itm_cheta, Rc=np.inf),
        values.E_itm_cheta[:, None],
    ) * values.P_itm_cheta
    return IFO + CHETA

def I_ITMY_HR(x):
    if values.out is not None:
        IFO = get_intensity(x, lho.ITMY.p1.i.q, values.out["E_itmy1"][:, None]) * 0.5e-6
    else:
        IFO = np.zeros_like(x[0])

    CHETA = get_intensity(
        x,
        finesse.BeamParam(w=values.w_itm_cheta, Rc=np.inf),
        values.E_itm_cheta[:, None],
    ) * values.P_itm_cheta

    return IFO + CHETA
    
def I_ETMX_HR(x):
    if values.out is not None:
        IFO = get_intensity(x, lho.ETMX.p1.i.q, values.out["E_etmx"][:, None]) * 0.5e-6 * 3 / 5
    else:
        IFO = np.zeros_like(x[0])

    CHETA = get_intensity(
        x,
        finesse.BeamParam(w=values.w_etm_cheta, Rc=np.inf),
        values.E_etm_cheta[:, None],
    ) * values.P_etm_cheta

    return IFO + CHETA

def I_ETMY_HR(x):
    if values.out is not None:
        IFO = get_intensity(x, lho.ETMY.p1.i.q, values.out["E_etmy"][:, None]) * 0.5e-6 * 3 / 5
    else:
        IFO = np.zeros_like(x[0])

    CHETA = get_intensity(
        x,
        finesse.BeamParam(w=values.w_etm_cheta, Rc=np.inf),
        values.E_etm_cheta[:, None],
    ) * values.P_etm_cheta

    return IFO + CHETA

# %%
values = SimpleNamespace()

print("Compute 1W RH steady state")
values.P_itm_RH = 1
values.P_etm_RH = 1
values.P_ACO2 = 0
values.P_itm_cheta = 0
values.w_itm_cheta = 53e-3
values.E_itm_cheta = np.zeros(lho.homs.shape[0])
values.P_etm_cheta = 0
values.w_etm_cheta = 62e-3
values.E_etm_cheta = np.zeros(lho.homs.shape[0])
values.out = None

ss_itm.temperature.I_HR.interpolate(I_ITMX_HR)
ss_itm.temperature.I_BR.interpolate(I_ITM_RH)
ss_itm.solve_temperature()
ss_itm.solve_deformation()
ITM_RH_SUB_1W = get_opd(x, y, ss_itm)
ITM_RH_SRF_1W = get_deformation(x, y, ss_itm)

ss_etm.temperature.I_HR.interpolate(I_ETMX_HR)
ss_etm.temperature.I_BR.interpolate(I_ETM_RH)
ss_etm.solve_temperature()
ss_etm.solve_deformation()
ETM_RH_SUB_1W = get_opd(x, y, ss_etm)
ETM_RH_SRF_1W = get_deformation(x, y, ss_etm)

# print("Compute 1W CHETA")
# values.P_itm_RH = 0
# values.P_etm_RH = 0
# values.P_ACO2 = 0
# values.P_itm_cheta = 1
# values.P_etm_cheta = 1

# ss_itm.temperature.I_HR.interpolate(I_ITMX_HR)
# ss_itm.temperature.I_BR.interpolate(I_ITM_RH)
# ITM_CHETA_SUB_1W = ss_itm.solve_temperature().copy()
# ITM_CHETA_SRF_1W = ss_itm.solve_deformation().copy()

# ss_etm.temperature.I_HR.interpolate(I_ETMX_HR)
# ss_etm.temperature.I_BR.interpolate(I_ETM_RH)
# ETM_CHETA_SUB_1W = ss_etm.solve_temperature().copy()
# ETM_CHETA_SRF_1W = ss_etm.solve_deformation().copy()

# %%
print("Compute 1W ITM ACO2")
I_ACO2, _ = ACO2.make_ACO2_intensity(values)

values.P_itm_RH = 0
values.P_etm_RH = 0
values.P_ACO2 = 1
ss_aco2 = ACO2.solve(CP, I_ACO2)
ACO2_OPD = get_opd(x, y, ss_aco2)
I_ACO2_interp = RegularGridInterpolator(
    (x, y),
    ACO2_OPD
)
# %%
# from ifo_thermal_state.fea import evaluate_solution

# points, surf, mask = evaluate_solution(
#     ss_itm.model.msh, x, y, 0.1, ITM_CHETA_SRF_1W, meshgrid=True
# )

# surf[~mask] = 0
# surf = surf[:, :, 0, 2]

# plt.imshow(surf)

# %%
include_RH_in_FEA = True
FEA_RESULT = {}

def initialise(values, initialise=False, run_locks=False, N=0):
    lho.L0.P = 2
    
    if initialise:
        values.out = None

        values.ACO2_suppression = 0
        values.P_ACO2 = 5

        values.P_itm_RH = 9
        values.P_itm_cheta = 0.4
        values.w_itm_cheta = 53e-3
        values.E_itm_cheta = np.zeros(lho.homs.shape[0])
        values.E_itm_cheta[0] = 1

        values.P_etm_RH = 6.5
        values.P_etm_cheta = 0.4
        values.w_etm_cheta = 62e-3
        values.E_etm_cheta = np.zeros(lho.homs.shape[0])
        values.E_etm_cheta[0] = 1

    ss_itm = AdvancedLIGOTestMass3DSteadyState(model)
    ss_itm.temperature.I_HR.interpolate(I_ITMX_HR)
    if include_RH_in_FEA:
        ss_itm.temperature.I_BR.interpolate(I_ITM_RH)
    else:
        ss_itm.temperature.I_BR.interpolate(I_ZERO)
    ss_itm.solve_temperature()
    #ss_itm.solve_deformation()

    ss_etm = AdvancedLIGOTestMass3DSteadyState(model)
    ss_etm.temperature.I_HR.interpolate(I_ETMX_HR)
    if include_RH_in_FEA:
        ss_etm.temperature.I_BR.interpolate(I_ETM_RH)
    else:
        ss_etm.temperature.I_BR.interpolate(I_ZERO)
    ss_etm.solve_temperature()
    #ss_etm.solve_deformation()

    lho.ITMYlens.f.value = base.ITMYlens.f.value
    lho.ITMXlens.f.value = base.ITMXlens.f.value

    lho.ITMY.Rc = base.ITMY.Rcx
    lho.ITMX.Rc = base.ITMX.Rcx
    lho.ETMY.Rc = base.ETMY.Rcx
    lho.ETMX.Rc = base.ETMX.Rcx

    # Set initial condition for steady state 
    ts_itmx.temperature.uh.interpolate(ss_itm.temperature.solution)
    ts_itmx.temperature.u_n.interpolate(ss_itm.temperature.solution)
    ts_itmy.temperature.uh.interpolate(ss_itm.temperature.solution)
    ts_itmy.temperature.u_n.interpolate(ss_itm.temperature.solution)

    ts_etmx.temperature.uh.interpolate(ss_etm.temperature.solution)
    ts_etmx.temperature.u_n.interpolate(ss_etm.temperature.solution)
    ts_etmy.temperature.uh.interpolate(ss_etm.temperature.solution)
    ts_etmy.temperature.u_n.interpolate(ss_etm.temperature.solution)

    # Do an initial step to get the displacement computed
    # temperature should be a steady state
    ts_itmx.temperature.I_HR.interpolate(I_ITMX_HR)
    if include_RH_in_FEA:
        ts_itmx.temperature.I_BR.interpolate(I_ITM_RH)
    else:
        ts_itmx.temperature.I_BR.interpolate(I_ZERO)

    ts_itmy.temperature.I_HR.interpolate(I_ITMY_HR)
    if include_RH_in_FEA:
        ts_itmy.temperature.I_BR.interpolate(I_ITM_RH)
    else:
        ts_itmy.temperature.I_BR.interpolate(I_ZERO)

    ts_etmx.temperature.I_HR.interpolate(I_ETMX_HR)
    if include_RH_in_FEA:
        ts_etmx.temperature.I_BR.interpolate(I_ETM_RH)
    else:
        ts_etmx.temperature.I_BR.interpolate(I_ZERO)
    
    ts_etmy.temperature.I_HR.interpolate(I_ETMY_HR)
    if include_RH_in_FEA:
        ts_etmy.temperature.I_BR.interpolate(I_ETM_RH)
    else:
        ts_etmy.temperature.I_BR.interpolate(I_ZERO)

    ts_itmx.step()
    ts_itmy.step()
    ts_etmx.step()
    ts_etmy.step()
    # Re-zero time
    ts_itmx.t = ts_etmx.t = ts_itmy.t = ts_etmy.t = 0

    update_results()
    update_maps()

    if values.ACO2_suppression > 0:
        # Get the left over junk in the substrate maps and save the negative of 
        # if, the 
        values.ACO2_suppression_data = -lho.ITMXlens.OPD_map.opd.copy()
        update_maps()

    sol = lho.run(InitialLockLIGO(run_locks=run_locks, exception_on_lock_fail=False, lock_steps=2000, gain_scale=0.4, pseudo_lock_arms=False))

    if run_locks:
        for _ in range(N):
            update_maps()
            lho.run("run_locks()")

    return sol

def update_results():
    global FEA_RESULT
    FEA_RESULT['ITMX_SRF'] = get_deformation(x, y, ts_itmx)
    FEA_RESULT['ITMY_SRF'] = get_deformation(x, y, ts_itmy)
    FEA_RESULT['ETMX_SRF'] = get_deformation(x, y, ts_etmx)
    FEA_RESULT['ETMY_SRF'] = get_deformation(x, y, ts_etmy)

    FEA_RESULT['ITMX_SUB'] = get_opd(x, y, ts_itmx)
    FEA_RESULT['ITMY_SUB'] = get_opd(x, y, ts_itmy)

def update_fea():
    global dt
    global dt_target
    global dt_diff
    global FEA_RESULT

    if abs(dt - dt_target) > 1e-3 and dt_diff == 0:
        dt_diff = 0.1 * (dt_target - dt)
    elif (dt >= dt_target and dt_diff > 0) or (dt <= dt_target and dt_diff < 0):
        dt_diff = 0
        dt = dt_target

    dt += dt_diff

    if ts_itmx.t > 100 and lho.L0.P.value < 25:
        print("Turning up power to 25")
        dt = 1

        values.P_itm_cheta *= (1-25/140)
        values.P_etm_cheta *= (1-25/140)

        lho.L0.P = 25
        lho.DARM_rf_lock.gain *= 2/25
        lho.CARM_lock.gain /= 25/2
        lho.PRCL_lock.gain /= 25/2
        lho.SRCL_lock.gain /= 25/2
        lho.MICH_lock.gain /= 25/2

    elif ts_itmx.t > 100 + 5 * 60 and lho.L0.P.value < 140:
        print("Turning up power to 140")
        lho.L0.P = 140
        dt = 1
        values.P_itm_cheta = 0
        values.P_etm_cheta = 0
        lho.CARM_lock.gain /= 140/25
        lho.DARM_rf_lock.gain /= 140/25
        lho.PRCL_lock.gain /= 140/25
        lho.SRCL_lock.gain /= 140/25
        lho.MICH_lock.gain /= 140/25

    ts_itmx.dt.value = ts_etmx.dt.value = dt
    ts_itmy.dt.value = ts_etmy.dt.value = dt

    ts_itmx.temperature.I_HR.interpolate(I_ITMX_HR)
    
    if include_RH_in_FEA:
        ts_itmx.temperature.I_BR.interpolate(I_ITM_RH)
    else:
        ts_itmx.temperature.I_BR.interpolate(I_ZERO)

    ts_itmy.temperature.I_HR.interpolate(I_ITMY_HR)
    if include_RH_in_FEA:
        ts_itmy.temperature.I_BR.interpolate(I_ITM_RH)
    else:
        ts_itmy.temperature.I_BR.interpolate(I_ZERO)

    ts_etmx.temperature.I_HR.interpolate(I_ETMX_HR)
    if include_RH_in_FEA:
        ts_etmx.temperature.I_BR.interpolate(I_ETM_RH)
    else:
        ts_etmx.temperature.I_BR.interpolate(I_ZERO)
    
    ts_etmy.temperature.I_HR.interpolate(I_ETMY_HR)
    if include_RH_in_FEA:
        ts_etmy.temperature.I_BR.interpolate(I_ETM_RH)
    else:
        ts_etmy.temperature.I_BR.interpolate(I_ZERO)

    ts_itmx.step()
    ts_etmx.step()
    ts_itmy.step()
    ts_etmy.step()

    update_results()

def update_maps(N=3):
    global FEA_RESULT
    for ts, TM, static, P_RH, S_RH in zip(
            [ts_itmx, ts_etmx, ts_itmy, ts_etmy],
            [lho.ITMX, lho.ETMX, lho.ITMY, lho.ETMY],
            [ITMX_static, ETMX_static, ITMY_static, ETMY_static],
            [values.P_itm_RH, values.P_etm_RH, values.P_itm_RH,  values.P_etm_RH],
            [ITM_RH_SRF_1W, ETM_RH_SRF_1W, ITM_RH_SRF_1W, ETM_RH_SRF_1W]
        ):
        TM.surface_map = Map(
            x,
            y,
            amplitude=TM_MASK,
            opd=FEA_RESULT[TM.name+'_SRF'] + (P_RH * S_RH if not include_RH_in_FEA else 0),
        )
        a, _ = TM.surface_map.get_radius_of_curvature_reflection(TM.p1.i.qx.w)
        TM.Rc = 2/(2/(base.get(TM.Rcx)) + 2/a)
        #plt.figure()
        #plt.title(TM.name)
        #plt.plot(TM.surface_map.opd[:, 100])
        TM.surface_map.remove_curvatures(TM.p1.i.qx.w)
        TM.surface_map.remove_tilts(TM.p1.i.qx.w)
        TM.surface_map.remove_piston(TM.p1.i.qx.w)
        #print(f'{TM.name} Rc: {TM.Rc} dioptre: {2/a/1e-6} [uD]')
        #plt.plot(TM.surface_map.opd[:, 100])
        #print(TM.name, TM.surface_map.opd.min(), TM.surface_map.opd.max())
        
    for TM, ts, LENS, P_RH, S_RH in zip(
            [lho.ITMX, lho.ITMY],
            [ts_itmx, ts_itmy],
            [lho.ITMXlens, lho.ITMYlens],
            [values.P_itm_RH, values.P_itm_RH],
            [ITM_RH_SUB_1W, ITM_RH_SUB_1W]
        ):

        LENS.OPD_map = Map(
            x, y,
            amplitude=X_aperture,
            opd=FEA_RESULT[TM.name+'_SUB'] + values.P_ACO2 * I_ACO2_interp((x, y)) + (P_RH * S_RH if not include_RH_in_FEA else 0),
        )
        # Extract the focal length in the map
        f_map = LENS.OPD_map.get_thin_lens_f(
            LENS.p1.i.qx.w, average=True
        )
        #print(LENS, f_map)
        if abs(f_map) != np.inf:
            LENS.f.value = 1/(1/base.get(LENS.f) + 1/f_map)
        else:
            LENS.f.value = base.get(LENS.f)

        LENS.OPD_map.opd[:] += values.ACO2_suppression * values.ACO2_suppression_data
        #plt.figure()
        #plt.title(f"{LENS.name} f={LENS.f.value}")
        #plt.plot(LENS.OPD_map.opd[:, 100])
        LENS.OPD_map.remove_curvatures(LENS.p1.i.qx.w, mode="average")
        LENS.OPD_map.remove_tilts(LENS.p1.i.qx.w)
        LENS.OPD_map.remove_piston(LENS.p1.i.qx.w)
        #plt.plot(LENS.OPD_map.opd[:, 100])
        #print(LENS.name, LENS.p1.i.qx.w, LENS.OPD_map.opd.min(), LENS.OPD_map.opd.max())

    lho.beam_trace()

# %%
include_RH_in_FEA = False
lho = base_lho.deepcopy()
lho.modes("even", maxtem=14)
lho.POP9.output_detectors = True
lho.POP45.output_detectors = True
lho.REFL9.output_detectors = True
lho.AS45.output_detectors = True
lho.beam_trace()
values = SimpleNamespace()

values.P_itm_RH = 10
values.P_etm_RH = 6.5 * 3/5
values.P_ACO2 = 0
values.ACO2_suppression = 0.95
values.ACO2_suppression_data = 0.0
values.P_itm_cheta = 0.4
values.w_itm_cheta = 53e-3
values.E_itm_cheta = np.zeros(lho.homs.shape[0])
values.E_itm_cheta[0] = 1
values.P_etm_cheta = 0.0 * 3/5
values.w_etm_cheta = 62e-3
values.E_etm_cheta = np.zeros(lho.homs.shape[0])
values.E_etm_cheta[0] = 1
values.out = None

init_sols = initialise(values, False, run_locks=True, N=2)

print("=================")
print("Initial")
DC(lho)
print("=================")

# %%
#init_sols['run locks'].plot_error_signals()

# %%
from copy import deepcopy
import gpstime

t = [0]
models = [lho.deepcopy()]
outs = [lho.run()]
values.out = outs[0]
locks = []
applied_values = []

dt = 20
dt_target = 20
dt_diff = 0
start_time = int(gpstime.gpsnow())

while ts_itmx.t <= 4000:
    print(dt, dt_target, dt_diff)
    update_fea()
    t.append(ts_itmx.t)
    applied_values.append(deepcopy(values))

    update_maps()
    lho.run(fac.SetLockGains(gain_scale=0.4))
    models.append(lho.deepcopy())
    sols = lho.run("series(run_locks(exception_on_fail=False, max_iterations=2000), noxaxis())")
    locks.append(sols['run locks'])
    values.out = sols["noxaxis"]
    outs.append(values.out)
    #eigensols.append(sols["eigenmode"])
    
    print(ts_itmx.t, sols["noxaxis"]["Parm"], sols["noxaxis"]["PRG"],  lho.ITMXlens.f.value,  lho.ITMYlens.f.value)



# %%
from pathlib import Path
import pickle

path = Path("./data/")
path.mkdir(exist_ok=True)
out = path / f"cheta_time_{str(int(start_time))}.pkl"
print(out)
with open(out, "wb") as file:
    pickle.dump(
        {
            "outs": outs,
            "t": t,
            "applied_values": applied_values,
            "models": [m.unparse() for m in models],
        },
        file,
    )

# %%
for m in models:
    m.beam_trace()

# %%
fig, axs = plt.subplots(3, 2, sharex=True, figsize=(11, 9))

N = len(outs)
plt.suptitle(f"{(1-values.ACO2_suppression)*100}% substrate residual reduction")
plt.sca(axs[0,0])
plt.plot(t[:N], tuple(out['PRG9'] for out in outs), label='9')
plt.plot(t[:N], tuple(out['PRG'] for out in outs), label='Carrier')
plt.ylabel("PRG")
plt.legend()

plt.sca(axs[1,0])
PCHETA_ITM = np.array([v.P_itm_cheta for v in applied_values])
PCHETA_ETM = np.array([v.P_etm_cheta for v in applied_values])
PRH_ITM = np.array([v.P_itm_RH for v in applied_values])
PRH_ETM = np.array([v.P_etm_RH for v in applied_values])
plt.plot(t[:N], tuple(out['Pin'] for out in outs), label='Input')
plt.plot(t[:(N-1)], 100 * PCHETA_ITM, label='50 x CHETA ITM')
plt.plot(t[:(N-1)], 100 * PCHETA_ETM, label='50 x CHETA ETM', ls='-.', lw=2)
plt.plot(t[:(N-1)], 10 * PRH_ITM, label='5 x RH ITM', ls='--')
plt.plot(t[:(N-1)], 10 * PRH_ETM, label='5 x RH ETM', ls='--')
plt.legend()
plt.title("Actuation")

plt.sca(axs[2,0])
plt.plot(t[:N], tuple(1/m.ITMXlens.f/1e-6 for m in models), label='ITM lens')
plt.ylabel("Power [uD]")
plt.legend()

plt.sca(axs[0, 1])
PX = np.array([out['Px'] for out in outs])
PY = np.array([out['Py'] for out in outs])
plt.plot(t[:N], PX, label='x arm')
plt.plot(t[:N], PY, label='y arm')
plt.ylabel("Power [W]")
plt.legend()
plt.title("Arm power")

plt.sca(axs[1, 1])
plt.plot(t[:N], PX*0.5e-6, label='X IFO')
plt.plot(t[:(N-1)], PCHETA_ITM, label='CHETA')
plt.plot(t[:(N-1)], PX[:-1]*0.5e-6 + PCHETA_ITM, label='Total', ls='--')
plt.ylabel("Power [W]")
plt.legend()
plt.title("ITM Absorbed power")


plt.sca(axs[2,1])
plt.plot(t[:N], tuple(m.ITMX.p1.i.qx.w/1e-3 for m in models), label='X arm')
plt.plot(t[:N], tuple(m.ITMY.p1.i.qx.w/1e-3 for m in models), label='Y arm', ls='--')
plt.hlines(values.w_itm_cheta/1e-3, 0, max(t[:N]), ls='--', label='CHETA', color='k')
plt.ylabel("spot size [mm]")
plt.title("ITM spot size")

plt.sca(axs[2,0])
plt.xlabel("Time [s]")
plt.sca(axs[2,1])
plt.xlabel("Time [s]")
plt.legend()
plt.tight_layout()
plt.savefig(f"figures/ITM_and_ETM_time_{int(start_time)}.pdf")





 # %%
plt.plot(t, tuple(out['AGX'] for out in outs), label='X arm')
plt.plot(t, tuple(out['AGY'] for out in outs), label='Y arm')
plt.ylabel("PRG")
plt.xlabel("Time [s]")
plt.title("Uniform coating absorption")
plt.legend()

# %%
plt.plot(t[:-1], tuple(out['Prefl'] for out in outs))
plt.ylabel("Power [W]")
plt.xlabel("Time [s]")
plt.title("REFL")

# %%
plt.plot(t, tuple(out['Pas_c'] for out in outs))
plt.ylabel("Power [W]")
plt.xlabel("Time [s]")
plt.title("AS_C")

# %%
plt.plot(t, tuple(out['Px']/1e3 for out in outs), label='X arm')
plt.plot(t, tuple(out['Py']/1e3 for out in outs), label='Y arm', ls='--')
plt.ylabel("Power [kW]")
plt.xlabel("Time [s]")
plt.title("Uniform coating absorption")
plt.legend()

# %%
plt.plot(t[:-1], tuple(m.ITMX.p1.i.qx.w/1e-3 for m in models), label='X arm')
plt.plot(t[:-1], tuple(m.ITMY.p1.i.qx.w/1e-3 for m in models), label='Y arm', ls='--')

plt.ylabel("ITM carrier spot size [mm]")
plt.xlabel("Time [s]")
plt.title("Uniform coating absorption")
plt.legend()

# %%
plt.plot(t[:-1], tuple(m.ITMXAR.p1.i.qx.w/1e-3 for m in models), label='X arm')
plt.plot(t[:-1], tuple(m.ITMYAR.p1.i.qx.w/1e-3 for m in models), label='Y arm', ls='--')

plt.ylabel("PRX carrier spot size [mm]")
plt.xlabel("Time [s]")
plt.title("Uniform coating absorption")
plt.legend()

# %%
for m in models:
    m.beam_trace()

# %%
plt.semilogy(t, tuple(abs(out['E_prc_l9']) for out in outs))
plt.ylabel("Power [kW]")
plt.xlabel("Time [s]")
plt.title("Uniform coating absorption")
plt.legend()

# %%
HOM_fraction = lambda E: np.sum(abs(E[:, 1:])**2, axis=1)/abs(E[:, 0])**2

plt.semilogy(t, HOM_fraction(np.array(tuple(out['E_prc_c0'] for out in outs))), label='c0')
plt.semilogy(t, HOM_fraction(np.array(tuple(out['E_prc_u9'] for out in outs))), label='u9')
plt.semilogy(t, HOM_fraction(np.array(tuple(out['E_prc_l9'] for out in outs))), label='l9')
plt.semilogy(t, HOM_fraction(np.array(tuple(out['E_prc_u45'] for out in outs))), label='u45')
plt.semilogy(t, HOM_fraction(np.array(tuple(out['E_prc_l45'] for out in outs))), label='l45')
plt.ylabel("HOM fraction [fractional]")
plt.xlabel("Time [s]")
plt.title("PRC HOM fraction\nUniform coating absorption")
plt.legend()

# %%
plt.semilogy(t, HOM_fraction(np.array(tuple(out['E_src_u9'] for out in outs))), label='u9')
plt.semilogy(t, HOM_fraction(np.array(tuple(out['E_src_l9'] for out in outs))), label='l9')
plt.semilogy(t, HOM_fraction(np.array(tuple(out['E_src_u45'] for out in outs))), label='u45')
plt.semilogy(t, HOM_fraction(np.array(tuple(out['E_src_l45'] for out in outs))), label='l45')
plt.ylabel("HOM fraction [fractional]")
plt.xlabel("Time [s]")
plt.title("SRC HOM fraction\nUniform coating absorption")
plt.legend()

# %%
E = lambda x: np.array(tuple(out[x] for out in outs))
HG20_fraction = lambda E: abs(E[:, lho.mode_index_map[2, 0]])**2/abs(E[:, 0])**2
HG02_fraction = lambda E: abs(E[:, lho.mode_index_map[0, 2]])**2/abs(E[:, 0])**2

plt.semilogy(t, HG20_fraction(E('E_inx_c0')), label='c0')
plt.semilogy(t, HG02_fraction(E('E_inx_c0')), label='c0')

plt.semilogy(t, HG20_fraction(E('E_inx_u9')), label='u9')
plt.semilogy(t, HG02_fraction(E('E_inx_u9')), label='u9')
plt.semilogy(t, HG20_fraction(E('E_inx_l9')), label='l9')
plt.semilogy(t, HG02_fraction(E('E_inx_l9')), label='l9')
# %%
