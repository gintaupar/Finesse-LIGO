# %%
import finesse
from finesse.knm import Map
import finesse.analysis.actions as fac
from finesse.ligo.factory import ALIGOFactory
from finesse.ligo.maps import (get_test_mass_surface_profile_interpolated, 
                               aligo_O4_TM_aperture, aligo_O4_ESD_inner_aperture,
                               aligo_O4_PR3_SR3_baffle)
from finesse.ligo.actions import InitialLockLIGO
from matplotlib.backends.backend_pdf import PdfPages
import gpstime
import pickle
from copy import deepcopy
from finesse.cymath.homs import HGModes
import numpy as np
from thermal_rom import *
from types import SimpleNamespace
import time
import h5py
import sys
sys.path.append("../analysis/O4/LLO/drmi/")
from aligo_newbs import ALIGOBeamSplitterFactory
from finesse.materials import FusedSilica
from tqdm import tqdm

# %%
def llo_O4_HR_baffle(r_lim = 0.21, N = 200, AoI = 45):
    x = np.linspace(-r_lim, r_lim, N)
    X, Y = np.meshgrid(x, x)
    inches_2_m = 25.4e-3
    r_major = 11.69 / 2 * inches_2_m * np.cos(np.deg2rad(AoI))
    r_minor = 10.236 / 2 * inches_2_m
    x_offset =   1.433 * inches_2_m * np.cos(np.deg2rad(AoI)) 
    
    # in the coordinat of BS:
    ellipse_main = ((X - x_offset+ 37.5e-3)/ r_major) ** 2 + (Y / r_minor) ** 2 
    ellipse_offset = ((X + x_offset+37.5e-3)/ r_major) ** 2 + (Y / r_minor) ** 2 
    bs_hr_baffle = np.ones_like(X)
    bs_hr_baffle[np.logical_and(ellipse_main > 1, ellipse_offset > 1)] = 0
    
    return x, bs_hr_baffle

def llo_O4_BS_AR_baffle(r_lim = 0.21, N = 200, AoI = 45, offset_direction = -1):
    x  = np.linspace(-r_lim, r_lim, N)
    X, Y = np.meshgrid(x, x)
    inches_2_m = 25.4e-3

    # Use Snell law to compute offset from input
    xoffset_out = offset_direction *( 80e-3 - (60e-3 * np.tan(np.arcsin((1/np.sqrt(2))/1.4996))))
    x_offset = offset_direction * 2.74 * inches_2_m * np.cos(np.deg2rad(AoI))   
    r_major = 11.69 / 2 * inches_2_m * np.cos(np.deg2rad(AoI))
    r_minor = 10.236 / 2 * inches_2_m
    ellipse_main = ((X - x_offset + xoffset_out)/ r_major) ** 2 + (Y / r_minor) ** 2 
    ellipse_offset = ((X + x_offset + xoffset_out)/ r_major) ** 2 + (Y / r_minor) ** 2 
    bs_ar_baffle = np.ones_like(X)
    bs_ar_baffle[np.logical_and(ellipse_main > 1, ellipse_offset > 1)] = 0
    return x, bs_ar_baffle

def llo_O4_BS_HR_aperture(r_lim = 0.24, N=301, AoI = 45):
    x  = np.linspace(-r_lim, r_lim, N)
    X, Y = np.meshgrid(x, x)
    a_bs = 0.185
    r_major = a_bs 
    r_minor = a_bs * np.cos(np.deg2rad(AoI))
    bs_ellipse= (X / r_minor) ** 2 + (Y / r_major) ** 2
    bs_aperture = np.ones_like(X)
    bs_aperture[bs_ellipse > 1] = 0
    return x, bs_aperture

def llo_O4_BS_AR_aperture(r_lim = 0.24, N=301, 
                          AoI=np.arcsin(1/np.sqrt(2)/FusedSilica.nr),
                          offset = -1):
    x  = np.linspace(-r_lim, r_lim, N)
    X, Y = np.meshgrid(x, x)
    a_bs = 0.185
    r_major = a_bs 
    r_minor = a_bs * np.cos(np.deg2rad(AoI))
    t_bs = 60e-3
    x_offset = offset * t_bs * np.tan(AoI)
    bs_ellipse= ((X+x_offset) / r_minor) ** 2 + (Y / r_major) ** 2
    bs_aperture = np.ones_like(X)
    bs_aperture[bs_ellipse > 1] = 0
    return x, bs_aperture

def ifo_model(new_bs = True, add_BS_baffle = True, update_parameters = True):
    if new_bs:
        factory = ALIGOBeamSplitterFactory(finesse.ligo.git_path() / "LLO" / "yaml" / "llo_O4.yaml")
    else:
        factory = ALIGOFactory(finesse.ligo.git_path() / "LLO" / "yaml" / "llo_O4.yaml")
    factory.update_parameters(finesse.ligo.git_path() / "LLO" / "yaml" / "llo_addRH.yaml")
    factory.params.INPUT.LASER.power = 2 
    factory.options.INPUT.add_IMC_and_IM1 = False
    factory.options.LSC.add_output_detectors = True
    factory.options.ASC.add = False
    factory.options.thermal.add = True
    factory.options.LSC.add_locks = True   
    if update_parameters:
        factory.update_parameters('../analysis/O4/LLO/drmi/llo_cold_state.yaml')
    model = factory.make()
    if new_bs:
        if add_BS_baffle: 
            xBS, BS_HRPR_aperture = llo_O4_HR_baffle()  
            _, BS_HRY_aperture = llo_O4_HR_baffle()
            xBSARX, BS_ARX_aperture = llo_O4_BS_AR_baffle(offset_direction=-1) 
            xBSARAS, BS_ARAS_aperture = llo_O4_BS_AR_baffle(offset_direction=1)

            model.BSHRPR.surface_map = Map(xBS, xBS, amplitude = BS_HRPR_aperture) 
            model.BSHRY.surface_map = Map(xBS, xBS, amplitude = BS_HRY_aperture)
            model.BSARX.surface_map = Map(xBSARX, xBSARX, amplitude = BS_ARX_aperture) 
            model.BSARAS.surface_map = Map(xBSARAS, xBSARAS, amplitude = BS_ARAS_aperture)    
        else:
            xBS, BS_HR_aperture = llo_O4_BS_HR_aperture()
            xBSARX, BS_ARX_aperture = llo_O4_BS_AR_aperture(offset_direction=-1) 
            xBSARAS, BS_ARAS_aperture = llo_O4_BS_AR_aperture(offset_direction=1)
            model.BS.surface_map = Map(xBS, xBS, amplitude = BS_HR_aperture) 
            model.BSARX.surface_map = Map(xBSARX, xBSARX, amplitude = BS_ARX_aperture) 
            model.BSARAS.surface_map = Map(xBSARAS, xBSARAS, amplitude = BS_ARAS_aperture)
    xPR3_SR3, PR3_SR3_aperture = aligo_O4_PR3_SR3_baffle()
    model.PR3.surface_map = Map(xPR3_SR3, xPR3_SR3, amplitude = PR3_SR3_aperture)
    model.SR3.surface_map = Map(xPR3_SR3, xPR3_SR3, amplitude = PR3_SR3_aperture)

    if update_parameters:
        model.remove(model.gIM2_p1_i)
        model.add(
            finesse.components.Gauss(
                'g1',
                model.PRMAR.p2.i,
                **factory.params.beam_waist,
            )
        )   
    return factory, model

# %% Load in the parameter file to make the 
# factory = ALIGOFactory(finesse.ligo.git_path() / "LLO" / "yaml" / "llo_O4.yaml")
# factory.update_parameters(finesse.ligo.git_path() / "LLO" / "yaml" / "llo_addRH.yaml")
# factory.options.INPUT.add_IMC_and_IM1 = False
# factory.options.LSC.add_output_detectors = True
# factory.options.ASC.add = False
# factory.options.thermal.add = True
# factory.options.LSC.add_locks = True
# factory.params.INPUT.LASER.power = 2
factory, llo = ifo_model(new_bs=True, 
                         add_BS_baffle= True, 
                         update_parameters=True)
# Make thermal models:
ts_itmx, ts_etmx, ts_itmy, ts_etmy = make_ts_optics()
# %% Make the ifo model
# factory.reset() # always reset to default
# factory.options.LSC.add_output_detectors = True
# factory.options.ASC.add = True
add_thermal_detectors(llo) 

# ring heater powers [W] with 70% efficiency from requested power to optic
rh_efficiency = 1
P_RH_ITMX = 1.0 
P_RH_ITMY = 0.8 
P_RH_ETMX = 2.2 
P_RH_ETMY = 2.4 

def set_ringheaters(arm, P_RH_ITM, P_RH_ETM):
    lens = llo.get(f"ITM{arm}lens")
    lens.f = 1 / (1 / lens.f + P_RH_ITM * factory.params.IRH_sub)
    itm = llo.get(f"ITM{arm}")
    etm = llo.get(f"ETM{arm}")
    itm.Rc = 2 / (2 / itm.Rc + P_RH_ITM * factory.params.IRH_srf)
    etm.Rc = 2 / (2 / etm.Rc + P_RH_ETM * factory.params.ERH_srf)

# set_ringheaters("X", P_RH_ITMX, P_RH_ETMX)
# set_ringheaters("Y", P_RH_ITMY, P_RH_ETMY)

base = llo.deepcopy()
initial_params = {p.full_name: p.value for p in base.all_parameters}

# %% Run cold state model to set starting point
# Set apertures for test mass:

R = 0.17
N = 201
x, TM_aperture = aligo_O4_TM_aperture(R, N)
x, lens_aperture = aligo_O4_ESD_inner_aperture(R, N)
y = x

# Get surfaces
for TM in [llo.ITMX, llo.ITMY, llo.ETMX, llo.ETMY]:
    if 'X' in TM.name:
        P = factory.params.X
    else:
        P = factory.params.Y
    TM._unfreeze()
    TM.static = get_test_mass_surface_profile_interpolated(P[TM.name[:-1]].ID, make_axisymmetric=True)(x, y)
    TM.aperture = TM_aperture
    TM._freeze()
    
    TM.surface_map = Map(x, y, amplitude = TM.aperture, opd=TM.static)

for TMlens in [llo.ITMXlens, llo.ITMYlens]:
    TMlens._unfreeze()
    TMlens.aperture = lens_aperture
    TMlens._freeze()
    TMlens.OPD_map = Map(x, y, amplitude = TMlens.aperture)

# llo.ITMX.surface_map = Map(x, y, amplitude=get_mask(x, y, r_ap = R), opd=llo.ITMX.static)
# llo.ETMX.surface_map = Map(x, y, amplitude=get_mask(x, y, r_ap = R), opd=llo.ETMX.static)
# llo.ITMY.surface_map = Map(x, y, amplitude=get_mask(x, y, r_ap = R), opd=llo.ITMY.static)
# llo.ETMY.surface_map = Map(x, y, amplitude=get_mask(x, y, r_ap = R), opd=llo.ETMY.static)
# llo.ITMXlens.OPD_map = Map(x, y, amplitude=get_mask(x, y, r_ap = R))
# llo.ITMYlens.OPD_map = Map(x, y, amplitude=get_mask(x, y, r_ap = R))

# compute the round trip losses with the maps in and make sure overall loss is reasonable
llo.modes( maxtem=8)
eigx = llo.run("eigenmodes(cavXARM, 0)")
eigy = llo.run("eigenmodes(cavYARM, 0)")

loss_x = (llo.X_arm_loss + eigx.loss(True)[1][0])
loss_y = (llo.Y_arm_loss + eigy.loss(True)[1][0])
print("X arm loss: ", loss_x/1e-6, "ppm")
print("Y arm loss: ", loss_y/1e-6, "ppm")
# Apply corrections to get back to original losses
print("Old X arm plane-wave loss: ", llo.X_arm_loss/1e-6, "ppm")
print("Old Y arm plane-wave loss: ", llo.Y_arm_loss/1e-6, "ppm")
llo.X_arm_loss -= eigx.loss(True)[1][0]
llo.Y_arm_loss -= eigy.loss(True)[1][0]
print("New X arm plane-wave loss: ", llo.X_arm_loss/1e-6, "ppm")
print("New Y arm plane-wave loss: ", llo.Y_arm_loss/1e-6, "ppm")
# %%
lock = InitialLockLIGO(
    exception_on_lock_fail=False,
    lock_steps=100,
    gain_scale=0.4,
    pseudo_lock_arms=False,
    run_locks=True
)
sol = llo.run(lock)

# save initial gains to be changed during power up
initial_gains = {}
for lsc_dof in ['DARM_rf', 'CARM', 'PRCL', 'SRCL', 'MICH']:
    initial_gains[lsc_dof] = llo.get(f'{lsc_dof}_lock.gain')

# %%
sol_llo = llo.run(
    fac.Series(
        lock,
        fac.Noxaxis(name="noxaxis"),
        fac.DCFields(name="dcfield")
    )
)
print(
    sol_llo['noxaxis']['Px'],
    sol_llo['noxaxis']['PRG'],
    sol_llo['noxaxis']['PRG9'],
    sol_llo['noxaxis']['AGX'],
    sol_llo['noxaxis']['AGY'],
)
# %%
rh_effs = np.linspace(0, 0.9, 20)
for rh_eff in tqdm(rh_effs):
    _P_RH_EX = rh_eff * P_RH_ETMX
    _P_RH_EY = rh_eff * P_RH_ETMY
    _P_RH_IX = rh_eff * P_RH_ITMX
    _P_RH_IY = rh_eff * P_RH_ITMY
    for ETM, P_ERH in zip((llo.ETMX, llo.ETMY), (_P_RH_EX, _P_RH_EY)):
        ETM.Rc =  2 / (2 / float(initial_params[ETM.Rcx.full_name]) + 
                    P_ERH * llo.ERH_srf.eval() )
    for ITMlens, ITM, P_RH_TM in zip((llo.ITMXlens, llo.ITMYlens),
                            (llo.ITMX, llo.ITMY),
                            (_P_RH_IX, _P_RH_IY )):
        ITMlens.f.value = 1 / (1 / float(initial_params[ITMlens.f.full_name]) 
                            + P_RH_TM * llo.IRH_sub.eval())
        ITM.Rc =  2 / (2 / float(initial_params[ITM.Rcx.full_name]) + 
                    P_RH_TM * llo.IRH_srf.eval() )
        
        llo.beam_trace()
        sol = llo.run(
            fac.Series(
                fac.RunLocks(
                            max_iterations=2000, 
                            exception_on_fail=False,
                            display_progress=True,
                            ),
                fac.SetLockGains(gain_scale=0.5)
            )
        )
# %%
sol_llo = llo.run(
    fac.Series(
        lock,
        fac.Noxaxis(name="noxaxis"),
        fac.DCFields(name="dcfield")
    )
)
print(
    sol_llo['noxaxis']['Px'],
    sol_llo['noxaxis']['PRG'],
    sol_llo['noxaxis']['PRG9'],
    sol_llo['noxaxis']['AGX'],
    sol_llo['noxaxis']['AGY'],
)
# %%
# Extract new initial parameters to run
initial_params_rh = {p.full_name: p.value for p in llo.all_parameters}

# %%
# update thermal state with rom:
itmx_absorption = 0.4e-6 ##alog70725    
itmy_absorption = 0.3e-6 #
etmx_absorption = 0.32e-6 #
etmy_absorption = 0.14e-6 #

def update_ts():
    if ts_itmx.t == 0:
        llo.ITMYlens.f.value = base.ITMYlens.f.value
        llo.ITMXlens.f.value = base.ITMXlens.f.value
    
    if ts_itmx.t > 180 and llo.L0.P != 25 and llo.L0.P < 25:
            llo.L0.P = 25
            llo.DARM_rf_lock.gain *= 2 / 25
            llo.CARM_lock.gain /= 25 / 2
            llo.PRCL_lock.gain /= 25 / 2
            llo.SRCL_lock.gain /= 25 / 2
            llo.MICH_lock.gain /= 25 / 2
    
    elif ts_itmx.t > 180 + 10 * 64 and llo.L0.P != 64:
            llo.L0.P = 64
            llo.CARM_lock.gain /= 64 / 25
            llo.DARM_rf_lock.gain /= 64 / 25
            llo.PRCL_lock.gain /= 64 / 25
            llo.SRCL_lock.gain /= 64 / 25
            llo.MICH_lock.gain /= 64 / 25
    
    # Update intensity:
    if ts_itmx.t < 1000:
        ts_itmx.dt = ts_etmx.dt = 20
        ts_itmy.dt = ts_etmy.dt = 20
    elif ts_itmx.t < 1500:
        ts_itmx.dt = ts_etmx.dt = 100
        ts_itmy.dt = ts_etmy.dt = 100
    elif ts_itmx.t >= 1500:
        ts_itmx.dt = ts_etmx.dt = 200
        ts_itmy.dt = ts_etmy.dt = 200

    for hr_func, _ts_optic, _abs in zip([I_ITMX_HR, I_ETMX_HR, I_ITMY_HR, I_ETMY_HR], 
                                  [ts_itmx, ts_etmx, ts_itmy, ts_etmy],
                                  [itmx_absorption, etmx_absorption, itmy_absorption, etmy_absorption]):
        u_k, I_abs = hr_func(llo, values, absorption =  _abs)
        _ts_optic.I.append(I_abs)
        _ts_optic.uI.append(u_k)
        compute_new_opd_state(_ts_optic)
        compute_new_deformation_state(_ts_optic)
        _ts_optic.t += _ts_optic.dt


    
# %%
initial_P = 2
llo.L0.P = initial_P
values = SimpleNamespace()
values.out = None
values.x = x
values.y = y
t = [0]
models = [llo.deepcopy()]
llo.beam_trace()
outs = [llo.run()]
values.out = outs[0]
eigensols = []
locks = []
data_time = int(gpstime.gpsnow())
values_used = []

program_starts = time.time()
while ts_itmx.t <= 10000:
    print(ts_itmx.t)
    values_used.append(deepcopy(values))
    update_ts()
    t.append(ts_itmx.t)

    update_maps(initial_params_rh, llo, values, ts_itmx, ts_etmx, ts_itmy, ts_etmy)
    llo.run(fac.SetLockGains(gain_scale=0.4))
    models.append(llo.deepcopy())
    
    sols = llo.run("series(run_locks(exception_on_fail=False, max_iterations=500), noxaxis())")
    locks.append(sols['run locks'])
    values.out = sols["noxaxis"]
    outs.append(values.out)

    print(ts_itmx.t, sols["noxaxis"]["Parm"], sols["noxaxis"]["PRG"],  llo.ITMXlens.f.value,  llo.ITMYlens.f.value)
end_times = time.time()
print(f'Elapsed time: {end_times - program_starts} s')


# %%
from pathlib import Path
import gpstime
import pickle

path = Path("./figures/")
path.mkdir(exist_ok=True)
out = path / f"llo_power_up_{data_time}_high_absorption.pkl"
print(out)
with open(out, "wb") as file:
    pickle.dump(
        {
            "parameters": factory.params,
            "options": factory.options,
            "outs": outs,
            "t": t,
            #"models": [m.unparse() for m in models],
        },
        file,
    )
# %%
finesse.init_plotting()

with PdfPages(f'figures/llo_power_up_{data_time}.pdf') as pdf:
    plt.plot(t, tuple(out['Pin'] for out in outs))
    plt.ylabel("Power [W]")
    plt.xlabel("Time [s]")
    plt.title("Input power")
    pdf.savefig()
    plt.close()

    plt.plot(t, tuple(out['Prefl']/1e-3 for out in outs))
    plt.ylabel("Power [mW]")
    plt.xlabel("Time [s]")
    plt.title("REFL")
    pdf.savefig()
    plt.close()

    plt.plot(t, tuple(out['Px']/1e3 for out in outs), label='X arm')
    plt.plot(t, tuple(out['Py']/1e3 for out in outs), label='Y arm', ls='--')
    plt.ylabel("Power [kW]")
    plt.xlabel("Time [s]")
    plt.title("Arm power")
    plt.legend()
    pdf.savefig()
    plt.close()

    plt.plot(t, tuple(out['AGX'] for out in outs), label='X arm')
    plt.plot(t, tuple(out['AGY'] for out in outs), label='Y arm')
    plt.ylabel("Gain")
    plt.xlabel("Time [s]")
    plt.title("Arm gains")
    plt.legend()
    pdf.savefig()
    plt.close()

    plt.plot(t, tuple(out['PRG9'] for out in outs), label='9')
    plt.plot(t, tuple(out['PRG45'] for out in outs), label='45')
    plt.plot(t, tuple(out['PRG'] for out in outs), label='Carrier')
    plt.ylabel("Gain")
    plt.xlabel("Time [s]")
    plt.title("Recycling gains")
    plt.legend()
    pdf.savefig()
    plt.close()

    plt.plot(t, tuple(out['PRG'] for out in outs), label='Carrier')
    plt.ylabel("Gain")
    plt.xlabel("Time [s]")
    plt.title("Recycling gains")
    plt.legend()
    pdf.savefig()
    plt.close()

    plt.plot(t, tuple(np.sum(out['E_prc_u9'] * out['E_prc_l9'].conjugate() + out['E_prc_u9'].conjugate() * out['E_prc_l9']).real/out['Pin'] for out in outs))
    plt.ylabel("Power [arb.]")
    plt.xlabel("Time [s]")
    plt.title("RF18 / P_IN")
    pdf.savefig()
    plt.close()

    plt.plot(t, tuple(np.sum(out['E_prc_u45'] * out['E_prc_l45'].conjugate() + out['E_prc_u45'].conjugate() * out['E_prc_l45']).real/out['Pin'] for out in outs))
    plt.ylabel("Power [arb.]")
    plt.xlabel("Time [s]")
    plt.title("RF90 / P_IN")
    pdf.savefig()
    plt.close()

# %%
