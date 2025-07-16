# %%
import finesse
import finesse.ligo
from finesse.knm import Map
from finesse.cymath.homs import HGModes
import numpy as np
import finesse.materials
from scipy.interpolate import interp1d
from ifo_thermal_state.aligo_3D import (
    make_test_mass_model,
    AdvancedLIGOTestMass3DTime,
    AdvancedLIGOTestMass3DSteadyState
)
from ifo_thermal_state.mesh import sweep_cylinder_3D
from types import SimpleNamespace
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from tools import get_deformation, get_opd

import os
os.environ['CC'] = 'clang'
os.environ['CXX'] = 'clang++'
finesse.init_plotting()

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
        "num_elements": [3, 5, 15],
        "heights": heights,
        "add_flats": True,
        "HR_mesh_size": 0.02,
        "AR_mesh_size": 0.02,
        "mesh_algorithm": 6,
    },
)

# Make the fea models
ts_itmx = AdvancedLIGOTestMass3DTime(model, 30)

def initial_condition(x):
    return np.full((x.shape[1],), 0)

ts_itmx.set_initial_condition(initial_condition)

# %%
lho = finesse.Model()

# %%
# Loading in ring heater realistic angular distribution
I_RH_theta = np.loadtxt("./RH_intensity.dat", delimiter=",")
theta_RH, _idx = np.unique(I_RH_theta[:, 0], return_index=True)
I_RH_data = I_RH_theta[:, 1][_idx]
#plt.plot(theta_RH, I_RH_data)
I_RH_theta_interp = interp1d(
    theta_RH, I_RH_data, kind="linear", fill_value="extrapolate"
)

A_RH = 2 * np.pi * model.radius * w_RH

def I_RH(x):
    I = (
        I_RH_theta_interp(np.arctan2(x[1], x[0]) + np.pi)
        * np.double(
            (x[2] >= -model.thickness / 2 + z_RH - w_RH / 2)
            & (x[2] <= -model.thickness / 2 + z_RH + w_RH / 2)
        )
        / A_RH
    )
    return I * values.P_RH

def get_intensity(x, q, E):
    HGs = HGModes(q, lho.homs)
    a = HGs.compute_points(x[0], x[1]) * E
    E = np.sum(a, axis=0)
    return (E * E.conj()).real

# Get intensity on HR surface:
def I_ITMX_HR(x):
    if values.out is not None:
        IFO = get_intensity(x, finesse.BeamParam(w=values.w_ifo, Rc=np.inf), values.out["E_itmx1"][:, None]) * 0.5e-6
    else:
        IFO = np.array([0])

    CHETA = get_intensity(
        x,
        finesse.BeamParam(w=values.w_cheta, Rc=np.inf),
        values.E_cheta[:, None],
    ) * values.P_cheta
    return IFO + CHETA

# %%
R = 0.17
N = 101

x, y = (
    np.linspace(-R, R, N),
    np.linspace(-R, R, N),
)

lho.modes("even", maxtem=8)
values = SimpleNamespace()
values.out = None

values.P_RH = 0
values.P_cheta = 1
values.w_cheta = 62e-3 #53e-3
values.w_ifo = 62e-3 #53e-3
values.E_cheta = np.zeros(lho.homs.shape[0])
values.E_cheta[0] = 1

# %%
ss_itmx = AdvancedLIGOTestMass3DSteadyState(model)
ss_itmx.temperature.I_HR.interpolate(I_ITMX_HR)
ss_itmx.temperature.I_BR.interpolate(I_RH)
ss_itmx.solve_temperature()
ss_itmx.solve_deformation()

ss_HR_surf = get_deformation(x, y, ss_itmx)
ss_OPD = get_opd(x, y, ss_itmx)

# %%
HR_surf = []
OPD = []
t = [0]

ts_itmx = AdvancedLIGOTestMass3DTime(model, 3600)
ts_itmx.set_initial_condition(initial_condition)
# interpolate steady state solution over u field
ts_itmx.temperature.u_n.interpolate(ss_itmx.temperature.solution)
ts_itmx.temperature.uh.interpolate(ss_itmx.temperature.solution)
ts_itmx.t = 0

HR_surf.append(get_deformation(x, y, ts_itmx))
OPD.append(get_opd(x, y, ts_itmx))

ts_itmx.temperature.I_HR.interpolate(I_ITMX_HR)
ts_itmx.temperature.I_BR.interpolate(I_RH)

while ts_itmx.t <= 3600*4:
    print(ts_itmx.t)
    ts_itmx.step()
    t.append(ts_itmx.t)
    HR_surf.append(get_deformation(x, y, ts_itmx))
    OPD.append(get_opd(x, y, ts_itmx))

# %%
HR_surf = np.array(HR_surf)
OPD = np.array(OPD)
t = np.array(t)

# %%
ts_itmx = AdvancedLIGOTestMass3DTime(model, 3600)
ts_itmx.t = 0
ts_itmx.set_initial_condition(initial_condition)

full_OPD = []
full_HR_surf = []
t2 = [0]

full_HR_surf.append(get_deformation(x, y, ts_itmx))
full_OPD.append(get_opd(x, y, ts_itmx))

ts_itmx.temperature.I_HR.interpolate(I_ITMX_HR)
ts_itmx.temperature.I_BR.interpolate(I_RH)

while ts_itmx.t <= 3600*16:
    print(ts_itmx.t)
    ts_itmx.step()
    t2.append(ts_itmx.t)
    full_HR_surf.append(get_deformation(x, y, ts_itmx))
    full_OPD.append(get_opd(x, y, ts_itmx))

# %%
full_HR_surf = np.array(full_HR_surf)
full_OPD = np.array(full_OPD)
t2 = np.array(t2)

# %%
ss_HR_surf = get_deformation(x, y, ss_itmx)
ss_OPD = get_opd(x, y, ss_itmx)

# %%
def get_dioptres(t, HR_surf):
    surface_diopters = np.zeros_like(t, dtype=float)

    for i, _t in enumerate(t):
        mmap = finesse.knm.Map(x, y, opd=HR_surf[i])
        rx, ry = mmap.get_radius_of_curvature_reflection(values.w_ifo)
        surface_diopters[i] = 2/((rx+ry)/2)

    return surface_diopters

def get_substrate_dioptres(t, OPD):
    substrate_diopters = np.zeros_like(t, dtype=float)
    for i, _t in enumerate(t):
        mmap = finesse.knm.Map(x, y, opd=OPD[i])
        rx, ry = mmap.get_thin_lens_f(values.w_ifo)
        substrate_diopters[i] = 1/((rx+ry)/2)
    return substrate_diopters

# %%
surface_diopters = get_dioptres(t, HR_surf)
surface_diopters[0] = np.nan
plt.plot(t/3600 + 17, surface_diopters / 1e-6, c='b')
plt.scatter(t/3600 + 17, surface_diopters / 1e-6, c='b', label='Start from steady state')

surface_diopters = get_dioptres(t2, full_HR_surf)
plt.plot(t2/3600, surface_diopters / 1e-6, c='b')
plt.scatter(t2/3600, surface_diopters / 1e-6, c='b', label='Full', marker='s')

mmap = finesse.knm.Map(x, y, opd=ss_HR_surf)
rx, ry = mmap.get_radius_of_curvature_reflection(values.w_ifo)

plt.hlines(2/((rx+ry)/2)/1e-6, plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], ls='--', color='k', lw=2, label='steady-state')
plt.xlabel("Time [hr]")
plt.ylabel("Focal power [uD]")
plt.title("Surface")
plt.legend()

# %%
substrate_diopters = get_substrate_dioptres(t, OPD)
plt.plot(t/3600+17,  substrate_diopters/1e-6, c='r')
plt.scatter(t/3600+17,  substrate_diopters/1e-6, c='r', label='Start from steady state')

substrate_diopters = get_substrate_dioptres(t2, full_OPD)
plt.plot(t2/3600, substrate_diopters / 1e-6, c='r')
plt.scatter(t2/3600, substrate_diopters / 1e-6, c='r', label='Full', marker='s')

mmap = finesse.knm.Map(x, y, opd=ss_OPD)
rx, ry = mmap.get_thin_lens_f(values.w_ifo)
plt.hlines(1/((rx+ry)/2)/1e-6, plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], ls='--', color='k', lw=2, label='steady-state')
plt.xlabel("Time [hr]")
plt.ylabel("Focal power [uD]")
plt.title("Substrate")
plt.legend()
# %%
