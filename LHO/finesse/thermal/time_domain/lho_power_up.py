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

from ifo_thermal_state.aligo_3D import (
    make_test_mass_model,
    AdvancedLIGOTestMass3DTime,
)

from types import SimpleNamespace
from finesse.ligo.maps import get_test_mass_surface_profile_interpolated
import matplotlib.pyplot as plt
from tools import get_mask, get_deformation, get_opd

finesse.init_plotting()

# %% We first make a factory object that can generate an ALIGO model
# here we do so using the LHO O4 parameter file
factory = ALIGOFactory(finesse.ligo.git_path() / "LHO" / "yaml" / "lho_O4.yaml")
factory.update_parameters(finesse.ligo.git_path() / "LHO" / "yaml" / "lho_mcmc_RC_lengths.yaml")

# %%
add_point_absorber = True
model = make_test_mass_model(
    mesh_function_kwargs={
        "HR_mesh_size": 0.02,
        "AR_mesh_size": 0.03,
        "mesh_algorithm": 6,
        "point": (0.018, 0) if add_point_absorber else None,
    }
)
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
lho = factory.make()
lho.L0.P = 2
lho.parse("""
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
""")

base = lho.deepcopy()

#lho.ITMXlens.f = lho.ITMYlens.f

lho.parse("fd E_c0_as OM1.p1.i f=0")

R = 0.17
N = 201

x, y = (
    np.linspace(-R, R, N),
    np.linspace(-R, R, N),
)

# Get surfaces
ITMX_static = get_test_mass_surface_profile_interpolated(factory.params.X.ITM.ID, make_axisymmetric=False)(x, y)
ETMX_static = get_test_mass_surface_profile_interpolated(factory.params.X.ETM.ID, make_axisymmetric=False)(x, y)
ITMY_static = get_test_mass_surface_profile_interpolated(factory.params.Y.ITM.ID, make_axisymmetric=False)(x, y)
ETMY_static = get_test_mass_surface_profile_interpolated(factory.params.Y.ETM.ID, make_axisymmetric=False)(x, y)

# For test masses to always recompute, bit of a hack at the moment in FINESSE
lho.ITMX.misaligned.is_tunable = True
lho.ETMX.misaligned.is_tunable = True
lho.ITMY.misaligned.is_tunable = True
lho.ETMY.misaligned.is_tunable = True

lho.ITMX.surface_map = Map(x, y, amplitude=get_mask(x, y, ts_itmx), opd=ITMX_static)
lho.ETMX.surface_map = Map(x, y, amplitude=get_mask(x, y, ts_etmx), opd=ETMX_static)
lho.ITMY.surface_map = Map(x, y, amplitude=get_mask(x, y, ts_itmy), opd=ITMY_static)
lho.ETMY.surface_map = Map(x, y, amplitude=get_mask(x, y, ts_etmy), opd=ETMY_static)

lho.ITMXlens.OPD_map = Map(x, y, amplitude=get_mask(x, y, ts_itmx))
lho.ITMYlens.OPD_map = Map(x, y, amplitude=get_mask(x, y, ts_itmy))

# compute the round trip losses with the maps in and make sure overall loss
# is reasonable
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
lock = InitialLockLIGO(exception_on_lock_fail=False, lock_steps=100, gain_scale=0.4, pseudo_lock_arms=False)
sol = lho.run(lock)

# %%
DC = lho.run()
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

# %%
# define point absorber absoprtion coefficient
alpha_pa = 100e-6

# Get intensity on HR surface:
def I_ITMX_HR(x):
    HGs = HGModes(lho.ITMX.p1.i.q, lho.homs)
    a = HGs.compute_points(x[0], x[1]) * values.out["E_itmx1"][:, None]
    E = np.sum(a, axis=0)
    I = E * E.conj()

    if add_point_absorber:
        r = np.sqrt((x[0] - 0.018) ** 2 + (x[1] - 0) ** 2)
        pabs = np.zeros_like(r)
        pabs[r < 0.001] = 1
        return I.real * 0.5e-6 + pabs * I.real * alpha_pa
    else:
        return I.real * 0.5e-6
    
def I_ETMX_HR(x):
    HGs = HGModes(lho.ETMX.p1.i.q, lho.homs)
    a = HGs.compute_points(x[0], x[1]) * values.out["E_etmx"][:, None]
    E = np.sum(a, axis=0)
    I = E * E.conj()
    return I.real * 0.5e-6 * 3 / 5

def I_ETMY_HR(x):
    HGs = HGModes(lho.ETMY.p1.i.q, lho.homs)
    a = HGs.compute_points(x[0], x[1]) * values.out["E_etmy"][:, None]
    E = np.sum(a, axis=0)
    I = E * E.conj()
    return I.real * 0.5e-6 * 3 / 5

def I_ITMY_HR(x):
    HGs = HGModes(lho.ITMY.p1.i.q, lho.homs)
    a = HGs.compute_points(x[0], x[1]) * values.out["E_itmy1"][:, None]
    E = np.sum(a, axis=0)
    I = E * E.conj()
    return I.real * 0.5e-6 

def update_fea():
    if ts_itmx.t == 0:
        lho.ITMYlens.f.value = base.ITMYlens.f.value
        lho.ITMXlens.f.value = base.ITMXlens.f.value
        ts_itmx.set_initial_condition(initial_condition)
        ts_itmy.set_initial_condition(initial_condition)
        ts_etmx.set_initial_condition(initial_condition)
        ts_etmy.set_initial_condition(initial_condition)
    
    if ts_itmx.t > 180 and lho.L0.P != 25 and lho.L0.P < 25:
        ts_itmx.dt.value = ts_etmx.dt.value = 20
        ts_itmy.dt.value = ts_etmy.dt.value = 20
        lho.L0.P = 25
        lho.DARM_rf_lock.gain *= 2/25
        lho.CARM_lock.gain /= 25/2
        lho.PRCL_lock.gain /= 25/2
        lho.SRCL_lock.gain /= 25/2
        lho.MICH_lock.gain /= 25/2
    elif ts_itmx.t > 180 + 10 * 60 and lho.L0.P != 60:
        lho.L0.P = 60
        lho.CARM_lock.gain /= 60/25
        lho.DARM_rf_lock.gain /= 60/25
        lho.PRCL_lock.gain /= 60/25
        lho.SRCL_lock.gain /= 60/25
        lho.MICH_lock.gain /= 60/25
    
    if ts_itmx.t > 1000 and ts_itmx.t < 1500:
        ts_itmx.dt.value = ts_etmx.dt.value = 60
        ts_itmy.dt.value = ts_etmy.dt.value = 60
    elif ts_itmx.t >= 1500:
        ts_itmx.dt.value = ts_etmx.dt.value = 120
        ts_itmy.dt.value = ts_etmy.dt.value = 120

    ts_itmx.temperature.I_HR.interpolate(I_ITMX_HR)
    ts_etmx.temperature.I_HR.interpolate(I_ETMX_HR)
    ts_itmy.temperature.I_HR.interpolate(I_ITMY_HR)
    ts_etmy.temperature.I_HR.interpolate(I_ETMY_HR)

    ts_itmx.step()
    ts_etmx.step()
    ts_itmy.step()
    ts_etmy.step()

def update_maps():
    for ts, TM in zip([ts_itmx, ts_etmx, ts_itmy, ts_etmy], [lho.ITMX, lho.ETMX, lho.ITMY, lho.ETMY]):
        lho.ITMX.surface_map = Map(
            x,
            y,
            amplitude=get_mask(x, y, ts),
            opd=ITMX_static + get_deformation(x, y, ts),
        )
        a,_ = lho.ITMX.surface_map.get_radius_of_curvature_reflection(TM.p1.i.qx.w)
        TM.Rc = 2/(2/(base.get(TM.Rcx)) + 2/a)
        TM.surface_map.remove_curvatures(TM.p1.i.qx.w)
        TM.surface_map.remove_tilts(TM.p1.i.qx.w)
        print(f'{TM.name} Rc: {TM.Rc} dioptre: {2/a/1e-6} [uD]')
    

    lho.ITMXlens.OPD_map = Map(
        x, y, amplitude=get_mask(x, y, ts_itmx), opd=get_opd(x, y, ts_itmx)
    )
    lho.ITMXlens.f.value =  lho.ITMXlens.OPD_map.get_thin_lens_f(
        lho.ITMXlens.p1.i.qx.w, average=True
    )
    lho.ITMXlens.OPD_map.remove_curvatures(lho.ITMXlens.p1.i.qx.w, mode="average")
    lho.ITMXlens.OPD_map.remove_tilts(lho.ITMXlens.p1.i.qx.w)

    lho.ITMYlens.OPD_map = Map(
        x, y, amplitude=get_mask(x, y, ts_itmy), opd=get_opd(x, y, ts_itmy)
    )
    lho.ITMYlens.f.value =  lho.ITMYlens.OPD_map.get_thin_lens_f(
        lho.ITMYlens.p1.i.qx.w, average=True
    )
    lho.ITMYlens.OPD_map.remove_curvatures(lho.ITMYlens.p1.i.qx.w, mode="average")
    lho.ITMYlens.OPD_map.remove_tilts(lho.ITMYlens.p1.i.qx.w)
    
    lho.beam_trace()

# %%
lho.L0.P = 2
values = SimpleNamespace()
values.out = None
t = [0]
models = [lho.deepcopy()]
outs = [lho.run()]
values.out = outs[0]
eigensols = []
locks = []

while ts_itmx.t <= 10000:
    update_fea()
    t.append(ts_itmx.t)

    update_maps()
    lho.run(fac.SetLockGains(gain_scale=0.5))
    models.append(lho.deepcopy())
    
    sols = lho.run("series(run_locks(exception_on_fail=False, max_iterations=500), noxaxis())")
    locks.append(sols['run locks'])
    values.out = sols["noxaxis"]
    outs.append(values.out)
    #eigensols.append(sols["eigenmode"])

    print(ts_itmx.t, sols["noxaxis"]["Parm"], sols["noxaxis"]["PRG"],  lho.ITMXlens.f.value,  lho.ITMYlens.f.value)


# %%
from pathlib import Path
import gpstime
import pickle

path = Path("./data/")
path.mkdir(exist_ok=True)
out = path / f"lho_no_pabs_{str(int(gpstime.gpsnow()))}.pkl"
print(out)
with open(out, "wb") as file:
    pickle.dump(
        {
            "eigensols": eigensols,
            "outs": outs,
            "t": t,
            "add_point_absorber": add_point_absorber,
            "models": [m.unparse() for m in models],
        },
        file,
    )

# %%
plt.plot(t[:-1], tuple(out['PRG9'] for out in outs), label='9')
plt.plot(t[:-1], tuple(out['PRG'] for out in outs), label='Carrier')
plt.ylabel("PRG")
plt.xlabel("Time [s]")
plt.title("Uniform coating absorption")
plt.legend()

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
