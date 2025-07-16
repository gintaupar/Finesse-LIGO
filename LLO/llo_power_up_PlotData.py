# %%
#%load_ext autoreload

import finesse
from finesse.knm import Map
import finesse.analysis.actions as fac
from finesse.ligo.factory import ALIGOFactory
from finesse.ligo.maps import get_test_mass_surface_profile_interpolated
from finesse.ligo.actions import InitialLockLIGO
from matplotlib.backends.backend_pdf import PdfPages
import gpstime
import pickle
import numpy as np 
from copy import deepcopy

from thermal import *
#%aimport thermal

use_real_data = True # needs to be true to plot data
initial_data_time = 1400352800 #UTC	May 21, 2024 18:53:02UTC #1388643918 #TODO: add ability to specify time and pull data.

# %% Load in the parameter file to make the 
factory = ALIGOFactory(finesse.ligo.git_path() / "LLO" / "yaml" / "llo_O4.yaml")
factory.update_parameters(finesse.ligo.git_path() / "LLO" / "yaml" / "llo_addRH.yaml")
factory.params.INPUT.LASER.power = 2

# %% Make the FEA objects for solving
tm_model = make_model()
ts_itmx, ts_etmx, ts_itmy, ts_etmy = make_solvers(tm_model)

# %% Make the model
factory.reset() # always reset to default
factory.options.LSC.add_output_detectors = True
factory.options.ASC.add = True
llo = factory.make()
add_thermal_detectors(llo)

# ring heater powers [W] with 70% efficiency from requested power to optic
P_RH_ITMX = factory.params.P_RH_ITMX * 0.7
P_RH_ITMY = factory.params.P_RH_ITMY * 0.7
P_RH_ETMX = factory.params.P_RH_ETMX * 0.7
P_RH_ETMY = factory.params.P_RH_ETMY * 0.7

def set_ringheaters(arm, P_RH_ITM, P_RH_ETM):
    lens = llo.get(f"ITM{arm}lens")
    lens.f = 1 / (1 / lens.f + P_RH_ITM * factory.params.IRH_sub)
    itm = llo.get(f"ITM{arm}")
    etm = llo.get(f"ETM{arm}")
    itm.Rc = 2 / (2 / itm.Rc + P_RH_ITM * factory.params.IRH_srf)
    etm.Rc = 2 / (2 / etm.Rc + P_RH_ETM * factory.params.ERH_srf)

set_ringheaters("X", P_RH_ITMX, P_RH_ETMX)
set_ringheaters("Y", P_RH_ITMY, P_RH_ETMY)

base = llo.deepcopy()

# %% Run cold state model to set starting point
R = tm_model.radius
N = 201

x, y = (
    np.linspace(-R, R, N),
    np.linspace(-R, R, N),
)

# Get surfaces
for TM in [llo.ITMX, llo.ITMY, llo.ETMX, llo.ETMY]:
    if 'X' in TM.name:
        P = factory.params.X
    else:
        P = factory.params.Y
    TM._unfreeze()
    TM.static = get_test_mass_surface_profile_interpolated(P[TM.name[:-1]].ID, make_axisymmetric=True)(x, y)
    TM._freeze()

llo.ITMX.surface_map = Map(x, y, amplitude=get_mask(x, y, ts_itmx), opd=llo.ITMX.static)
llo.ETMX.surface_map = Map(x, y, amplitude=get_mask(x, y, ts_etmx), opd=llo.ETMX.static)
llo.ITMY.surface_map = Map(x, y, amplitude=get_mask(x, y, ts_itmy), opd=llo.ITMY.static)
llo.ETMY.surface_map = Map(x, y, amplitude=get_mask(x, y, ts_etmy), opd=llo.ETMY.static)

llo.ITMXlens.OPD_map = Map(x, y, amplitude=get_mask(x, y, ts_itmx))
llo.ITMYlens.OPD_map = Map(x, y, amplitude=get_mask(x, y, ts_itmy))

# compute the round trip losses with the maps in and make sure overall loss is reasonable
llo.modes("even", maxtem=8)
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

def get_data(chan_file: str = 'data.pkl'):
    '''
    To use real P_in data, in a terminal, first need to run
    >> python ligo-commissioning-modeling/scripts/data_munch.py L1_chans.yaml data.pkl
    for list of channels and times specified in L1_chans.yaml outputted to data.pkl
    can change the time interval in the L1_chans.yaml file
    '''
    with open(chan_file, 'rb') as f:
        data=pickle.load(f)
        return data

if use_real_data:
    print(f'Using real P_in data from IFO powerup at {initial_data_time}')
    d = get_data()
    Power_data = d[initial_data_time]['data']['L1:IMC-IM4_TRANS_SUM_OUTPUT'].value
    Power_data_dt = d[initial_data_time]['data']['L1:IMC-IM4_TRANS_SUM_OUTPUT'].dt.value

# %%
def update_fea():
    if ts_itmx.t == 0:
        llo.ITMYlens.f.value = base.ITMYlens.f.value
        llo.ITMXlens.f.value = base.ITMXlens.f.value
        ts_itmx.set_initial_condition(zero_initial_condition)
        ts_itmy.set_initial_condition(zero_initial_condition)
        ts_etmx.set_initial_condition(zero_initial_condition)
        ts_etmy.set_initial_condition(zero_initial_condition)
    
    if use_real_data:
        try:
            current_P = Power_data[int(ts_itmx.t/Power_data_dt)]
            llo.L0.P = current_P
        except IndexError:
            current_P = llo.L0.P # keep using last known P_in if data has ended. 
        print('Current time is {}s, power in is {}W.'.format(ts_itmx.t,current_P))

        # update gains
        for lsc_dof in ['DARM_rf', 'CARM', 'PRCL', 'SRCL', 'MICH']:
            obj = getattr(llo, f'{lsc_dof}_lock')
            obj.gain = initial_gains[lsc_dof] * initial_P / current_P

    else:
        # Using hardcoded P_in data for power up.
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

    if ts_itmx.t < 1000:
        ts_itmx.dt.value = ts_etmx.dt.value = 20
        ts_itmy.dt.value = ts_etmy.dt.value = 20
    elif ts_itmx.t < 1500:
        ts_itmx.dt.value = ts_etmx.dt.value = 100
        ts_itmy.dt.value = ts_etmy.dt.value = 100
    elif ts_itmx.t >= 1500:
        ts_itmx.dt.value = ts_etmx.dt.value = 200
        ts_itmy.dt.value = ts_etmy.dt.value = 200

    ts_itmx.temperature.I_HR.interpolate(partial(I_ITMX_HR, llo, values))
    ts_etmx.temperature.I_HR.interpolate(partial(I_ETMX_HR, llo, values))
    ts_itmy.temperature.I_HR.interpolate(partial(I_ITMY_HR, llo, values))
    ts_etmy.temperature.I_HR.interpolate(partial(I_ETMY_HR, llo, values))

    ts_itmx.step()
    ts_etmx.step()
    ts_itmy.step()
    ts_etmy.step()

# %%
initial_P = 2
llo.L0.P = initial_P
values = SimpleNamespace()
values.out = None
values.x = x
values.y = y
t = [0]
models = [llo.deepcopy()]
outs = [llo.run()]
values.out = outs[0]
eigensols = []
locks = []
data_time = int(gpstime.gpsnow())
values_used = []

while ts_itmx.t <= 10000:
    print(ts_itmx.t)
    values_used.append(deepcopy(values))
    update_fea()
    t.append(ts_itmx.t)

    update_maps(base, llo, values, ts_itmx, ts_etmx, ts_itmy, ts_etmy)
    llo.run(fac.SetLockGains(gain_scale=0.4))
    models.append(llo.deepcopy())
    
    sols = llo.run("series(run_locks(exception_on_fail=False, max_iterations=500), noxaxis())")
    locks.append(sols['run locks'])
    values.out = sols["noxaxis"]
    outs.append(values.out)

    print(ts_itmx.t, sols["noxaxis"]["Parm"], sols["noxaxis"]["PRG"],  llo.ITMXlens.f.value,  llo.ITMYlens.f.value)

# %%
from pathlib import Path
import gpstime
import pickle

path = Path("./figures/")
path.mkdir(exist_ok=True)
out = path / f"llo_power_up_{data_time}.pkl"
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

# %% Sorting data for plotting
P_in = d[initial_data_time]['data']['L1:IMC-IM4_TRANS_SUM_OUTPUT'].value
P_as = d[initial_data_time]['data']['L1:ASC-AS_C_SUM_OUTPUT'].value
P_refl = d[initial_data_time]['data']['L1:LSC-REFL_A_LF_OUTPUT'].value
P_pop = d[initial_data_time]['data']['L1:LSC-POP_A_LF_OUTPUT'].value
P_PRG = d[initial_data_time]['data']['L1:LSC-PRC_GAIN_MON'].value
Kappa_C = d[initial_data_time]['data']['L1:CAL-CS_TDEP_KAPPA_C_OUTPUT'].value
RF18 = d[initial_data_time]['data']['L1:LSC-POPAIR_B_RF18_I_MON'].value
RF90 = d[initial_data_time]['data']['L1:LSC-POPAIR_B_RF90_I_MON'].value
# To do, could change RF18 and RF90 to Magnitude =  SQRT(I^2 + Q^2)
t_data = np.linspace(0, len(P_in)*Power_data_dt, num = len(P_in))
   
# %%
finesse.init_plotting()

with PdfPages(f'figures/test_llo_power_up_{data_time}.pdf') as pdf:
    fig, ax1 = plt.subplots()
    plt.title("Input power")
    color = 'tab:blue'
    ax1.set_xlabel(f'Time after {initial_data_time} [s]')
    ax1.set_ylabel('Power [W]', color=color)
    ax1.plot(t, tuple(out['Pin'] for out in outs), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('L1:IMC-IM4_TRANS_SUM_OUTPUT', color=color)
    ax2.plot(t_data, P_in, label='data', color=color, alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)
    pdf.savefig()
    plt.close()

    fig, ax1 = plt.subplots()
    plt.title("REFL")
    color = 'tab:blue'
    ax1.set_xlabel(f'Time after {initial_data_time} [s]')
    ax1.set_ylabel('Power [mW]', color=color)
    ax1.plot(t, tuple(out['Prefl']/1e-3 for out in outs), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('L1:LSC-REFL_A_LF_OUTPUT', color=color)
    ax2.plot(t_data, P_refl, label='data', color=color, alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)
    pdf.savefig()
    plt.close()

    fig, ax1 = plt.subplots()
    plt.title("Pick-off PRC Power")
    color = 'tab:blue'
    ax1.set_xlabel(f'Time after {initial_data_time} [s]')
    ax1.set_ylabel('Power [mW] ??', color=color)
    ax1.plot(t, tuple(out['Ppop']/1e-3 for out in outs), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('L1:LSC-POP_A_LF_OUTPUT', color=color)
    ax2.plot(t_data, P_pop, label='data', color=color, alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)
    pdf.savefig()
    plt.close()

    # plt.plot(t, tuple(out['Px']/1e3 for out in outs), label='X arm')
    # plt.plot(t, tuple(out['Py']/1e3 for out in outs), label='Y arm', ls='--')
    # plt.ylabel("Power [kW]")
    # plt.xlabel("Time [s]")
    # plt.title("Arm power")
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    # plt.plot(t, tuple(out['AGX'] for out in outs), label='X arm')
    # plt.plot(t, tuple(out['AGY'] for out in outs), label='Y arm')
    # plt.ylabel("Gain")
    # plt.xlabel("Time [s]")
    # plt.title("")
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    fig, ax1 = plt.subplots()
    plt.title('Arm gains')
    color = 'tab:blue'
    ax1.set_xlabel(f'Time after {initial_data_time} [s]')
    ax1.set_ylabel('Gain', color=color)
    ax1.plot(t, tuple(out['AGX'] for out in outs), label='X arm', color=color)
    ax1.plot(t, tuple(out['AGY'] for out in outs), label='Y arm', color='orange')
    ax1.tick_params(axis='y', labelcolor=color)
    plt.legend()
    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('L1:CAL-CS_TDEP_KAPPA_C_OUTPUT', color=color)
    ax2.plot(t_data, Kappa_C, label='data', color=color, alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)
    pdf.savefig()
    plt.close()

    # plt.plot(t, tuple(out['PRG9'] for out in outs), label='9')
    # plt.plot(t, tuple(out['PRG45'] for out in outs), label='45')
    # plt.plot(t, tuple(out['PRG'] for out in outs), label='Carrier')
    # plt.ylabel("Gain")
    # plt.xlabel("Time [s]")
    # plt.title("Recycling gains")
    # plt.legend()
    # pdf.savefig()
    # plt.close()

    fig, ax1 = plt.subplots()
    plt.title('Recycling gains')
    color = 'tab:blue'
    ax1.set_xlabel(f'Time after {initial_data_time} [s]')
    ax1.set_ylabel('Gain', color=color)
    ax1.plot(t, tuple(out['PRG'] for out in outs), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('L1:LSC-PRC_GAIN_MON', color=color)
    ax2.plot(t_data, P_PRG, label='data', color=color, alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)
    pdf.savefig()
    plt.close()

    fig, ax1 = plt.subplots()
    plt.title("RF18 / P_IN")
    color = 'tab:blue'
    ax1.set_xlabel(f'Time after {initial_data_time} [s]')
    ax1.set_ylabel('Power [arb.]', color=color)
    ax1.plot(t, tuple(np.sum(out['E_prc_u9'] * out['E_prc_l9'].conjugate() + out['E_prc_u9'].conjugate() * out['E_prc_l9']).real/out['Pin'] for out in outs), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('POPAIR_B_RF18_I_MON / P_in', color=color)
    ax2.plot(t_data, [i / j for i, j in zip(RF18, P_in)], label='data', color=color, alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)
    pdf.savefig()
    plt.close()

    fig, ax1 = plt.subplots()
    plt.title("RF90 / P_IN")
    color = 'tab:blue'
    ax1.set_xlabel(f'Time after {initial_data_time} [s]')
    ax1.set_ylabel('Power [arb.]', color=color)
    ax1.plot(t, tuple(np.sum(out['E_prc_u45'] * out['E_prc_l45'].conjugate() + out['E_prc_u45'].conjugate() * out['E_prc_l45']).real/out['Pin'] for out in outs), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('POPAIR_B_RF90_I_MON / P_in', color=color)
    ax2.plot(t_data, [i / j for i, j in zip(RF90, P_in)], label='data', color=color, alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)
    pdf.savefig()
    plt.close()
# %%


# %%
