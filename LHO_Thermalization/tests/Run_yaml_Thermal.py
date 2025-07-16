# %%
import os
os.chdir('/Users/raeddiab/ligo-commissioning-modeling/analysis/O4/LHO_Thermalization/src')
import tqdm
import sys 
sys.path.append('/Users/raeddiab/ligo-commissioning-modeling/analysis/O4/LLO/drmi')
from aligo_newbs import ALIGOBeamSplitterFactory


import finesse
import numpy as np
import matplotlib.pyplot as plt
import gpstime, pickle, glob, os, sys
import finesse.analysis.actions as fac
from finesse.knm import Map
from finesse.ligo.factory import ALIGOFactory
from finesse.ligo.maps import get_test_mass_surface_profile_interpolated
from finesse.ligo.actions import InitialLockLIGO
from matplotlib.backends.backend_pdf import PdfPages
from copy import deepcopy
from finesse.ligo import maps
from funcs import (model_output_params,
                   reward_params,
                   param_variations,
                   get_astig_params,
                   run_thermal_model,
                   plot_thermal_evolution,
                   )

from thermal_rom import make_ts_optics, add_thermal_detectors

use_real_data = False
initial_data_time = 1418514918 # to do: add ability to specify time and pull data.
maxx = 6000

# %%
parentrepo = "/Users/raeddiab/ligo-commissioning-modeling"
repopath = f"{parentrepo}/analysis/O4/LHO_Thermalization"

sys.path.append(f'{parentrepo}/LLO')
sys.path.append(f'{parentrepo}/analysis/O4/LHO_Thermalization/src')

suffix = 'cold_state'

yamls = glob.glob(f'{repopath}/data/run_{suffix}/cold_state_yaml_solutions/*.yaml')
yamls.sort()

figfolder = f'{repopath}/figures/figs_{suffix}'
os.makedirs(figfolder, exist_ok=True)

factory = ALIGOBeamSplitterFactory(f"{parentrepo}/LHO/yaml/lho_O4.yaml")
factory.update_parameters(f"{parentrepo}/LHO/yaml/lho_addRH.yaml")
factory.update_parameters(yamls[0])
factory.update_parameters('/Users/raeddiab/ligo-commissioning-modeling/analysis/O4/LHO_Thermalization/data/run_cold_state/cold_state_yaml_solutions/cold_state_sol_0.yaml')
if 'absorption' in factory.params:
    tm_abs = factory.params.absorption

if 'RH_eff' in factory.params:
    RH_eff = factory.params.RH_eff

# %%
runsols = {}
for i, yaml in enumerate(yamls[:1]):
    if ('absorption' in factory.params) & ('RH_eff' in factory.params):
        t, outs, models, locks, _, _ = run_thermal_model(
                                    maxtems=8,
                                    datapath=f'{repopath}/data/run_{suffix}/cold_state_yaml_solutions/thermal_evolution_{suffix}_{i}.pkl',
                                    return_data_level=2,
                                    update_yaml=yaml,
                                    w0=1.015e-3,
                                    z=6.0,
                                    t_evol=maxx,
                                    runsols=runsols,
                                    model_output_params=model_output_params,
                                    tm_absorption=tm_abs,
                                    RH_efficiency=RH_eff,
                                    is_astig=True,
                                    use_real_data=True,
                                    repopath=parentrepo,
                                    new_bs=True,
                                    add_BS_baffle=True,
                                    initial_data_time=initial_data_time,
                                    )
    else:
            t, outs, models, locks, _, _ = run_thermal_model(
                                        maxtems=8,
                                        datapath=f'{repopath}/data/run_{suffix}/cold_state_yaml_solutions/thermal_evolution_{suffix}_{i}.pkl',
                                        return_data_level=2,
                                        update_yaml=yaml,
                                        w0=1.015e-3,
                                        z=6.0,
                                        t_evol=maxx,
                                        runsols=runsols,
                                        model_output_params=model_output_params,
                                        is_astig=True,
                                        )

# %%
fulldata = []

for i, yaml in enumerate(yamls[:1]):

    with open(f'{repopath}/data/run_{suffix}/cold_state_yaml_solutions/thermal_evolution_{suffix}_{i}.pkl', "rb") as file:
        data = pickle.load(file)

    fulldata.append({'t': data['t'], 'outs': data['outs'], 'varyParam': 'GA'})

plot_thermal_evolution(fulldata, {'param': f'YAML{i}'},
                       plotfileloc=f'{figfolder}/GA_state_powerup_{suffix}_{i}.pdf',
                       axislims=False)

print(f'Fig Saved to:\n{figfolder}/GA_state_powerup_{suffix}_{i}.pdf')



use_real_data = True # needs to be true to plot data
# initial_data_time = 1400352800
initial_data_time = 1418514918

import warnings 
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
initial_data_time =1418514918 # to do: add ability to specify time and pull data.
parentrepo='/Users/raeddiab/ligo-commissioning-modeling'
def get_data(chan_file: str = f'{parentrepo}/scripts/data.pkl'):
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
    d = get_data(f'{parentrepo}/scripts/data_{initial_data_time}.pkl')
    Power_data = d[initial_data_time]['data']['H1:IMC-IM4_TRANS_NSUM_OUT16'].value
    Power_data_dt = d[initial_data_time]['data']['H1:IMC-IM4_TRANS_NSUM_OUT16'].dt.value

# Sorting data for plotting
P_in = d[initial_data_time]['data']['H1:IMC-IM4_TRANS_NSUM_OUT16'].value
t_data = np.linspace(0, len(P_in)*Power_data_dt, num = len(P_in))
irange = np.where(t_data < maxx)

t_data = t_data[irange]
P_in = P_in[irange] * 1
P_as = d[initial_data_time]['data']['H1:ASC-AS_C_SUM_OUT16'].value[irange] / 6400 # 6.4 cts/mW
P_refl = d[initial_data_time]['data']['H1:LSC-REFL_A_LF_OUT16'].value[irange] /(0.25* 1e3) #accounting for 2 BSs from M5 to REFLs and converting to W
P_pop = d[initial_data_time]['data']['H1:LSC-POP_A_LF_OUT16'].value[irange] / (1e6*0.05)
P_PRG = d[initial_data_time]['data']['H1:LSC-PR_GAIN_OUT16'].value[irange] 
Kappa_C = d[initial_data_time]['data']['H1:CAL-CS_TDEP_KAPPA_C_OUTPUT'].value[irange] # normalized to 1
RF18 = d[initial_data_time]['data']['H1:LSC-POPAIR_B_RF18_I_MON'].value[irange]*1.17/P_in #* 64.7/2700 W/cts per input W
RF90 = d[initial_data_time]['data']['H1:LSC-POPAIR_B_RF90_I_MON'].value[irange]*0.7/P_in #* 27.8/700  W/cts per input W
# # To do, could change RF18 and RF90 to Magnitude =  SQRT(I^2 + Q^2)

factory = ALIGOFactory(f"{parentrepo}/LHO/yaml/lho_O4.yaml")

outs = fulldata[0]['outs']
t = np.array(fulldata[0]['t'])

irange1 = np.where(np.array(t) < maxx)

finesse.init_plotting()
lambda_op = lambda x, y, op: op(op(x), op(y))
tollow, tolhigh = 0.95, 1.05
ylim = 1
plotloc = f'{figfolder}/thermalization_{initial_data_time}_{suffix}.pdf'
print(plotloc)

save_plots = True  

if save_plots:
    with PdfPages(plotloc) as pdf:
        # Plot 1: Input power
        fig, ax1 = plt.subplots()
        plt.title("Input power")
        color = 'tab:blue'
        ax1.set_xlabel(f'Time after {initial_data_time} [s]')
        ax1.set_ylabel('Power [W]', color=color)

        simPin = np.array([out['Pin'] for out in outs])[irange1]
        miny, maxy = lambda_op(simPin, P_in, min), lambda_op(simPin, P_in, max)
        ax1.plot(t[irange1], simPin, color=color)
        if ylim: ax1.set_ylim(miny*tollow, maxy*tolhigh)
        ax1.set_xlim(0, maxx)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('H1:IMC-IM4_TRANS_SUM_OUTPUT', color=color)
        ax2.plot(t_data, P_in, label='data', color=color, alpha=0.6)
        if ylim: ax2.set_ylim(miny*tollow, maxy*tolhigh)
        ax2.tick_params(axis='y', labelcolor=color)
        if save_plots:
            pdf.savefig()
        plt.show()
        plt.close()

        # Plot 2: REFL
        fig, ax1 = plt.subplots()
        plt.title("REFL")
        color = 'tab:blue'
        ax1.set_xlabel(f'Time after {initial_data_time} [s]')
        ax1.set_ylabel('Power [W]', color=color)

        simPreflPRM = np.array([out['Prefl'] for out in outs])[irange1]
        miny, maxy = lambda_op(simPreflPRM, P_refl, min), lambda_op(simPreflPRM, P_refl, max)
        ax1.plot(t[irange1], simPreflPRM, color=color)
        if ylim: ax1.set_ylim(miny*tollow, maxy*tolhigh)
        ax1.set_xlim(0, maxx)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('H1:LSC-REFL_A_LF_OUTPUT', color=color)
        ax2.plot(t_data, P_refl, label='data', color=color, alpha=0.6)
        if ylim: ax2.set_ylim(miny*tollow, maxy*tolhigh)
        ax2.tick_params(axis='y', labelcolor=color)
        if save_plots:
            pdf.savefig()
        plt.show()
        plt.close()

        # Plot 3: Pick-off PRC Power
        fig, ax1 = plt.subplots()
        plt.title("Pick-off PRC Power")
        color = 'tab:blue'
        ax1.set_xlabel(f'Time after {initial_data_time} [s]')
        ax1.set_ylabel('Power [W]', color=color)
        simPpop = np.array([out['Ppop']for out in outs])[irange1]
        miny, maxy = lambda_op(simPpop, P_pop, min), lambda_op(simPpop, P_pop, max)
        ax1.plot(t[irange1], simPpop, color=color)
        if ylim: ax1.set_ylim(miny*tollow, maxy*tolhigh)
        ax1.set_xlim(0, maxx)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('H1:LSC-POP_A_LF_OUTPUT', color=color)
        ax2.plot(t_data, P_pop, label='data', color=color, alpha=0.6)
        if ylim: ax2.set_ylim(miny*tollow, maxy*tolhigh)
        ax2.tick_params(axis='y', labelcolor=color)
        if save_plots:
            pdf.savefig()
        plt.show()
        plt.close()

        # Plot 4: Arm gains
        fig, ax1 = plt.subplots()
        plt.title('Arm gains')
        color = 'tab:blue'
        ax1.set_xlabel(f'Time after {initial_data_time} [s]')
        ax1.set_ylabel('Gain', color=color)

        simAGX = np.array([out['AGX'] for out in outs])[irange1]
        ax1.plot(t[irange1], simAGX, color=color, label='X arm')

        simAGY = np.array([out['AGY'] for out in outs])[irange1]
        ax1.plot(t[irange1], simAGY, color='orange', label='Y arm')
        ax1.set_xlim(0, maxx)
        ax1.tick_params(axis='y', labelcolor=color)
        plt.legend()

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('H1:CAL-CS_TDEP_KAPPA_C_OUTPUT', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        if save_plots:
            pdf.savefig()
        plt.show()
        plt.close()

        # Plot 5: Recycling gains
        fig, ax1 = plt.subplots()
        plt.title('Recycling gains')
        color = 'tab:blue'
        ax1.set_xlabel(f'Time after {initial_data_time} [s]')
        ax1.set_ylabel('Gain', color=color)
        simPRG = np.array([out['PRG'] for out in outs])[irange1]
        miny, maxy = lambda_op(simPRG, P_PRG, min), lambda_op(simPRG, P_PRG, max)
        ax1.plot(t[irange1], simPRG, color=color)
        if ylim: ax1.set_ylim(miny*tollow, maxy*tolhigh)
        ax1.set_xlim(0, maxx)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('H1:LSC-PRC_GAIN_MON', color=color)
        ax2.plot(t_data, P_PRG, label='data', color=color, alpha=0.6)
        if ylim: ax2.set_ylim(miny*tollow, maxy*tolhigh)
        ax2.tick_params(axis='y', labelcolor=color)
        if save_plots:
            pdf.savefig()
        plt.show()
        plt.close()

        # Plot 6: PRG9
        fig, ax1 = plt.subplots()
        plt.title("PRG9")
        color = 'tab:blue'
        ax1.set_xlabel(f'Time after {initial_data_time} [s]')
        ax1.set_ylabel('Gain', color=color)
        simPRG9 = np.array([out['PRG9'] for out in outs])[irange1]
        miny, maxy = lambda_op(simPRG9, RF18, min), lambda_op(simPRG9, RF18, max)
        ax1.plot(t[irange1], simPRG9, color=color)
        if ylim: ax1.set_ylim(miny*tollow, maxy*tolhigh)
        ax1.set_xlim(0, maxx)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('POPAIR_B_RF18_I_MON', color=color)
        ax2.plot(t_data, RF18, label='data', color=color, alpha=0.6)
        if ylim: ax2.set_ylim(miny*tollow, maxy*tolhigh)
        ax2.tick_params(axis='y', labelcolor=color)
        if save_plots:
            pdf.savefig()
        plt.show()
        plt.close()

        # Plot 7: PRG45
        fig, ax1 = plt.subplots()
        plt.title("PRG45")
        color = 'tab:blue'
        ax1.set_xlabel(f'Time after {initial_data_time} [s]')
        ax1.set_ylabel('Power [arb.]', color=color)
        simPRG45 = np.array([out['PRG45'] for out in outs])[irange1]
        miny, maxy = lambda_op(simPRG45, RF90, min), lambda_op(simPRG45, RF90, max)
        ax1.plot(t[irange1], simPRG45, color=color)
        if ylim: ax1.set_ylim(miny*tollow, maxy*tolhigh)
        ax1.set_xlim(0, maxx)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('POPAIR_B_RF90_I_MON', color=color)
        ax2.plot(t_data, RF90, label='data', color=color, alpha=0.6)
        if ylim: ax2.set_ylim(miny*tollow, maxy*tolhigh)
        ax2.tick_params(axis='y', labelcolor=color)
        if save_plots:
            pdf.savefig()
        plt.show()
        plt.close()

# %%
