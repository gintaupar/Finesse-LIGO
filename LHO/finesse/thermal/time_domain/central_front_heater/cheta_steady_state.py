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
import gpstime
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
from pathlib import Path
import pickle

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

base_lho.POP9.output_detectors = True
base_lho.POP45.output_detectors = True
base_lho.REFL9.output_detectors = True
base_lho.AS45.output_detectors = True

base = base_lho.deepcopy()

R = 0.17
N = 201

x, TM_aperture = aligo_O4_TM_aperture(R, N)
x, X_aperture = aligo_O4_ESD_inner_aperture(R, N)
x, Y_aperture = aligo_O4_ESD_inner_aperture(R, N)
y = x
X, Y = np.meshgrid(x, y)

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

print("Compute 1W RH")
values.P_itm_RH = 1
values.P_etm_RH = 1
values.P_ACO2 = 0
values.P_itm_cheta = 0
values.w_itm_cheta = 53e-3
values.E_itm_cheta = np.zeros(lho.homs.shape[0])
values.E_itm_cheta[0] = 1
values.P_etm_cheta = 0
values.w_etm_cheta = 62e-3
values.E_etm_cheta = np.zeros(lho.homs.shape[0])
values.E_etm_cheta[0] = 1
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

print("Compute 1W CHETA")
values.P_itm_RH = 0
values.P_etm_RH = 0
values.P_ACO2 = 0
values.P_itm_cheta = 1
values.P_etm_cheta = 1

ss_itm.temperature.I_HR.interpolate(I_ITMX_HR)
ss_itm.temperature.I_BR.interpolate(I_ITM_RH)
ss_itm.solve_temperature()
ss_itm.solve_deformation()

ITM_CHETA_SUB_1W = get_opd(x, y, ss_itm)
ITM_CHETA_SRF_1W = get_deformation(x, y, ss_itm)

ss_etm.temperature.I_HR.interpolate(I_ETMX_HR)
ss_etm.temperature.I_BR.interpolate(I_ETM_RH)
ss_etm.solve_temperature()
ss_etm.solve_deformation()

ETM_CHETA_SUB_1W = get_opd(x, y, ss_etm)
ETM_CHETA_SRF_1W = get_deformation(x, y, ss_etm)
# %%
ITM_MASK = get_mask(x, y, ss_itm)
ETM_MASK = get_mask(x, y, ss_etm)

# %%
print("Compute 1W ITM ACO2")
I_ACO2, _ = ACO2.make_ACO2_intensity(values)

values.P_itm_RH = 0
values.P_etm_RH = 0
values.P_ACO2 = 1

ss_aco2 = AdvancedLIGOTestMass3DSteadyState(CP)
ss_aco2.temperature.I_HR.interpolate(I_ACO2)
ss_aco2.solve_temperature()

ACO2_SUB_1W = get_opd(x, y, ss_aco2)

# %%

# %%
def initialise(values, initialise=False, OPD_SCALE=1):
    lho.L0.P = 2
    
    if initialise:
        values.out = None
        
        values.P_ACO2 = 0

        values.P_itm_RH = 4.21052632
        values.P_itm_cheta = 0.4
        values.w_itm_cheta = 53e-3
        values.E_itm_cheta = np.zeros(lho.homs.shape[0])
        values.E_itm_cheta[0] = 1

        values.P_etm_RH = 6.5
        values.P_etm_cheta = 0.4
        values.w_etm_cheta = 62e-3
        values.E_etm_cheta = np.zeros(lho.homs.shape[0])
        values.E_etm_cheta[0] = 1

    lho.ITMYlens.f.value = base.ITMYlens.f.value
    lho.ITMXlens.f.value = base.ITMXlens.f.value

    lho.ITMY.Rc = base.ITMY.Rcx
    lho.ITMX.Rc = base.ITMX.Rcx
    lho.ETMY.Rc = base.ETMY.Rcx
    lho.ETMX.Rc = base.ETMX.Rcx
    update_maps(OPD_SCALE)
    return lho.run(InitialLockLIGO(run_locks=False, exception_on_lock_fail=False, lock_steps=2000, gain_scale=0.4, pseudo_lock_arms=False))

def update_maps(OPD_SCALE=1):
    for MASK, RH, CHETA, TM, static in zip(
            [ITM_MASK, ETM_MASK, ITM_MASK, ETM_MASK], 
            [values.P_itm_RH * ITM_RH_SRF_1W, values.P_etm_RH * ETM_RH_SRF_1W, values.P_itm_RH * ITM_RH_SRF_1W, values.P_etm_RH * ETM_RH_SRF_1W],
            [values.P_itm_cheta * ITM_CHETA_SRF_1W, values.P_etm_cheta * ETM_CHETA_SRF_1W, values.P_itm_cheta * ITM_CHETA_SRF_1W, values.P_etm_cheta * ETM_CHETA_SRF_1W],
            [lho.ITMX, lho.ETMX, lho.ITMY, lho.ETMY],
            [ITMX_static, ETMX_static, ITMY_static, ETMY_static],
        ):
        TM.surface_map = Map(
            x,
            y,
            amplitude=MASK,
            opd=RH + CHETA,
        )
        a, _ = TM.surface_map.get_radius_of_curvature_reflection(TM.p1.i.qx.w)
        TM.Rc = 2/(2/(base.get(TM.Rcx)) + 2/a)
        #plt.figure()
        #plt.title(TM.name)
        #plt.plot(TM.surface_map.opd[:, 100])
        TM.surface_map.remove_curvatures(TM.p1.i.qx.w)
        TM.surface_map.remove_tilts(TM.p1.i.qx.w)
        print(f'{TM.name} Rc: {TM.Rc} dioptre: {2/a/1e-6} [uD]')
        #plt.plot(TM.surface_map.opd[:, 100])
        
    for LENS in [lho.ITMXlens, lho.ITMYlens]:
        LENS.OPD_map = Map(
            x, y,
            amplitude=X_aperture,
            opd=(
                values.P_ACO2 * ACO2_SUB_1W +
                values.P_itm_RH * ITM_RH_SUB_1W + 
                values.P_itm_cheta * ITM_CHETA_SUB_1W
            ),
        )
        # Extract the focal length in the map
        f_map = LENS.OPD_map.get_thin_lens_f(
            LENS.p1.i.qx.w, average=True
        )
        print(LENS, f_map)
        if abs(f_map) != np.inf:
            LENS.f.value = 1/(1/base.get(LENS.f) + 1/f_map)
        else:
            LENS.f.value = base.get(LENS.f)

        #plt.figure()
        #plt.title(f"{LENS.name} f={LENS.f.value}")
        #plt.plot(LENS.OPD_map.opd[:, 100])
        LENS.OPD_map.remove_curvatures(LENS.p1.i.qx.w, mode="average")
        LENS.OPD_map.remove_tilts(LENS.p1.i.qx.w)
        LENS.OPD_map.opd[:] *= OPD_SCALE
        #plt.plot(LENS.OPD_map.opd[:, 100])

    lho.beam_trace()

def plot_operating_point(model):
    sol = model.run("""
        series(
            xaxis(CARM.DC, lin, -0.01, 0.01, 5, relative=True, name='CARM'),
            xaxis(DARM.DC, lin, -0.04, 0.04, 5, relative=True, name='DARM'),
            xaxis(MICH.DC, lin, -1, 1, 5, relative=True, name='MICH'),
            xaxis(PRCL.DC, lin, -1, 1, 5, relative=True, name='PRCL'),
            xaxis(SRCL.DC, lin, -10, 10, 5, relative=True, name='SRCL'),
        )""")

    sol_names = [s.name for s in sol.children]
    dof_lock = {
        lock.feedback.owner.name: lock for lock in model.locks if not lock.disabled
    }
    errors = []
    for dof in model.dofs:
        if dof.name in sol_names:
            error = sol[dof.name][dof_lock[dof.name].error_signal]
            error /= max(error)
            errors.append(error)

    errors = np.array(errors)
    plt.plot(errors.T, label=[
        "CARM",
        "DARM",
        "MICH",
        "PRCL",
        "SRCL",
    ])

def check_operating_point(model):
    sol = model.run("""
        series(
            xaxis(CARM.DC, lin, -0.005, 0.005, 2, relative=True, name='CARM'),
            xaxis(DARM.DC, lin, -0.02, 0.02, 2, relative=True, name='DARM'),
            xaxis(MICH.DC, lin, -0.5, 0.5, 2, relative=True, name='MICH'),
            xaxis(PRCL.DC, lin, -0.5, 0.5, 2, relative=True, name='PRCL'),
            xaxis(SRCL.DC, lin, -5, 5, 2, relative=True, name='SRCL'),
        )""")

    sol_names = [s.name for s in sol.children]
    dof_lock = {
        lock.feedback.owner.name: lock for lock in model.locks if not lock.disabled
    }
    errors = []
    for dof in model.dofs:
        if dof.name in sol_names:
            error = sol[dof.name][dof_lock[dof.name].error_signal]
            error /= max(error)
            errors.append(error)

    errors = np.array(errors)
    return all(abs(errors[:, 1]) < 0.9)

# %%
lho = base_lho.deepcopy()
lho.modes("even", maxtem=14)
lho.POP9.output_detectors = True
lho.POP45.output_detectors = True
lho.REFL9.output_detectors = True
lho.AS45.output_detectors = True
lho.beam_trace()
values = SimpleNamespace()

values.P_itm_RH = 9.5
values.P_etm_RH = 6.5
values.P_ACO2 = 0
values.P_itm_cheta = 0.4
values.w_itm_cheta = 53e-3
values.E_itm_cheta = np.zeros(lho.homs.shape[0])
values.E_itm_cheta[0] = 1

values.P_etm_cheta = 0.4
values.w_etm_cheta = 62e-3
values.E_etm_cheta = np.zeros(lho.homs.shape[0])
values.E_etm_cheta[0] = 1
values.out = None

init_sols = initialise(values, False)

DC(lho)
check_operating_point(lho)

# %%
def plot(data, name, start_time, title, xlabel, ylabel):
    X = np.asarray(data['X'])
    Y = np.asarray(data['Y'])

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    plt.sca(axs[0])
    Z = np.array(np.array([DC['PRG'] for DC in data['DCs']]))
    #Z[~np.array(DC3s['ok'])] = np.nan
    Z = Z.reshape((X.size, Y.size))

    def upscale(X, Y, Z):
        I = RegularGridInterpolator((X, Y), Z, method='cubic')
        _X = np.linspace(X.min(), X.max(), 20)
        _Y = np.linspace(Y.min(), Y.max(), 20)
        _XX, _YY = np.meshgrid(_X, _Y, indexing='ij')
        return _XX, _YY, I((_XX, _YY))
    
    _X, _Y, _Z = upscale(X, Y, Z)
    CS = plt.contour(_X, _Y, _Z, levels=[10, 20, 30, 40, 50, 55], colors='0.2', linestyles='--')
    plt.clabel(CS, CS.levels, inline=True, fmt="%i", fontsize=10)
    plt.contourf(_X, _Y, _Z,levels=30, cmap='Reds',vmin=0, vmax=60)

    plt.grid(False)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Carrier PRG")

    plt.sca(axs[1])
    Z = np.array(np.array([DC['PRG9'] for DC in data['DCs']]))
    #Z[~np.array(DC3s['ok'])] = np.nan
    Z = Z.reshape((X.size, Y.size))
    CS = plt.contour(X, Y, Z.T, levels=[10, 20, 30, 40, 50, 60, 70, 80], colors='0.2', linestyles='--')
    plt.clabel(CS, CS.levels, inline=True, fmt="%i", fontsize=10)
    plt.contourf(X,Y,Z.T,levels=30, cmap='Blues', vmin=0, vmax=85)
    plt.grid(False)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.title("9MHz PRG")

    plt.sca(axs[2])
    Z = np.array(np.array([DC['PRG45'] for DC in data['DCs']]))
    #Z[~np.array(DC3s['ok'])] = np.nan
    Z = Z.reshape((X.size, Y.size))
    CS = plt.contour(X, Y, Z.T, levels=[1, 2, 3, 4, 5, 6, 7, 8], colors='0.2', linestyles='--')
    plt.clabel(CS, CS.levels, inline=True, fmt="%i", fontsize=10)
    plt.contourf(X,Y,Z.T,levels=30, cmap='Greens', vmin=0, vmax=7)
    plt.grid(False)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.title("45MHz PRG")

    plt.suptitle(title)
    plt.tight_layout()

    plt.savefig(f"{name}_{int(start_time)}.pdf")

def save(data, number, start_time):
    path = Path("./data/")
    path.mkdir(exist_ok=True)
    out = path / f"cheta_steady_state_{int(number)}_{str(int(start_time))}.pkl"
    print(out)
    with open(out, "wb") as file:
        pickle.dump(
            {
                "DCs": data['DCs'],
                "X": data['X'],
                "Y": data['Y'],
                "mismatches": data['mismatches'],
                "ok": data['ok'],
                "models": [m.unparse() for m in data['models']],
            },
            file,
        )

# %%
DC2s = {}
DC2s['X'] = np.linspace(8, 11, 8)
DC2s['Y'] = np.linspace(0, 8, 9)
DC2s['DCs'] = []
DC2s['mismatches'] = []
DC2s['ok'] = []
DC2s['models'] = []
start_time = gpstime.gpsnow()

for XX in DC2s['X']:
    for YY in DC2s['Y']:
        lho = base_lho.deepcopy()
        lho.beam_trace()
        lho.modes("even", maxtem=14)
        
        values = SimpleNamespace()
        values.P_itm_RH = XX
        values.P_etm_RH = 6.5
        values.P_ACO2 = YY
        values.P_itm_cheta = 0.4
        values.w_itm_cheta = 53e-3
        values.E_itm_cheta = np.zeros(lho.homs.shape[0])
        values.E_itm_cheta[0] = 1
        values.P_etm_cheta = 0.4
        values.w_etm_cheta = 62e-3
        values.E_etm_cheta = np.zeros(lho.homs.shape[0])
        values.E_etm_cheta[0] = 1
        values.out = None  
        
        initialise(values, False)
        print("!!", XX, YY)
        _DC = lho.run()
        print("!!", "PRG", _DC['PRG'], "PRG9", _DC['PRG9'])

        DC2s['DCs'].append(_DC)
        DC2s['mismatches'].append(lho.cavity_mismatch())
        DC2s['ok'].append(check_operating_point(lho))
        DC2s['models'].append(lho)

# %%
save(DC2s, 2, start_time)
plot(DC2s, "current_aco2_suppression", start_time, "Cold (Pin=2W) steady state with 0.4W of CHETA on ITM & ETM]\nUsing as installed annular CO2 projection", "ITM RH [W]", "Annular CO2 [W]")

# %%
# run RH steps and just manually scale OPD to zero "ideal CO2 correction"
DC3s = {}
DC3s['X'] = np.linspace(8, 11, 5)
DC3s['Y'] = np.linspace(0, 1, 5)
DC3s['DCs'] = []
DC3s['mismatches'] = []
DC3s['ok'] = []
DC3s['models'] = []

for XX in DC3s['X']:
    for YY in DC3s['Y']:
        lho = base_lho.deepcopy()
        lho.beam_trace()
        lho.modes("even", maxtem=14)
        
        values = SimpleNamespace()
        values.P_itm_RH = XX
        values.P_etm_RH = 6.5
        values.P_ACO2 = 0
        values.P_itm_cheta = 0.4
        values.w_itm_cheta = 53e-3
        values.E_itm_cheta = np.zeros(lho.homs.shape[0])
        values.E_itm_cheta[0] = 1
        values.P_etm_cheta = 0.4
        values.w_etm_cheta = 62e-3
        values.E_etm_cheta = np.zeros(lho.homs.shape[0])
        values.E_etm_cheta[0] = 1
        values.out = None  
        
        initialise(values, False, OPD_SCALE=YY)
        print("!!", XX, YY)
        _DC = lho.run()
        print("!!", "PRG", _DC['PRG'], "PRG9", _DC['PRG9'])

        DC3s['DCs'].append(_DC)
        DC3s['mismatches'].append(lho.cavity_mismatch())
        DC3s['ok'].append(check_operating_point(lho))
        DC3s['models'].append(lho)

save(DC2s, 3, start_time)
plot(DC3s, "opd_suppression", start_time, "Cold (Pin=2W) steady state with 0.4W of CHETA on ITM & ETM]\nUsing idealised scaling of residual substrate OPD", "ITM RH [W]", "Annular CO2 [W]")

# %%
DC4s = {}
DC4s['X'] = np.linspace(0, 0.4, 20)
DC4s['Y'] = np.linspace(0, 9.5, 20)
DC4s['DCs'] = []
DC4s['mismatches'] = []
DC4s['ok'] = []
DC4s['models'] = []

for XX in DC4s['X']:
    for YY in DC4s['Y']:
        lho = base_lho.deepcopy()
        lho.beam_trace()
        lho.modes("even", maxtem=14)
        
        values = SimpleNamespace()
        values.P_itm_RH = YY
        values.P_etm_RH = 6.5 * XX/0.4
        values.P_ACO2 = 5 * XX/0.4
        values.P_itm_cheta = XX
        values.w_itm_cheta = 53e-3
        values.E_itm_cheta = np.zeros(lho.homs.shape[0])
        values.E_itm_cheta[0] = 1
        values.P_etm_cheta = XX
        values.w_etm_cheta = 62e-3
        values.E_etm_cheta = np.zeros(lho.homs.shape[0])
        values.E_etm_cheta[0] = 1
        values.out = None  
        
        initialise(values, False)
        print("!!", XX, YY)
        _DC = lho.run()
        print("!!", "PRG", _DC['PRG'], "PRG9", _DC['PRG9'])

        DC4s['DCs'].append(_DC)
        DC4s['mismatches'].append(lho.cavity_mismatch())
        DC4s['ok'].append(check_operating_point(lho))
        DC4s['models'].append(lho)

save(DC2s, 4, start_time)
plot(DC2s, "cheta_vs_rh_2", start_time, "Cold (Pin=2W) steady state with 0.4W of CHETA on ITM & ETM]", "ITM RH [W]", "CHETA [W]")
