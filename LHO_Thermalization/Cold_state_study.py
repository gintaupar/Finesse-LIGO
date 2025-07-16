# %%
%cd /Users/raeddiab/ligo-commissioning-modeling

import finesse
from finesse.ligo.factory import ALIGOFactory
from finesse.ligo.actions import InitialLockLIGO, DARM_RF_to_DC
from finesse.plotting import bode
import finesse.analysis.actions as fac
import finesse.analysis.actions as fa
from finesse.ligo.factory import aligo
import matplotlib.pylab as plt
import sympy as sym
from scipy.integrate import quad
from finesse.ligo.ASC_controllers import get_controller
from finesse.ligo.suspension import QUADSuspension
from scipy.integrate import *
import scipy
import finesse.components as fc
from IPython.display import Image
import finesse.detectors as det
from finesse.analysis.actions import FrequencyResponse
import cmath
from finesse.analysis.actions import (
    SensingMatrixDC,
    OptimiseRFReadoutPhaseDC)
from sympy import integrate
from finesse.analysis.actions.axes import Noxaxis
finesse.init_plotting()
from control import tf, root_locus
import sys
import matplotlib as mpl
import numpy as np
from finesse.symbols import Constant
import pandas as pd 
from IPython.display import Image
import gpstime, pickle, glob, os, sys
import yaml
import tqdm
import sys 
sys.path.append(f'{finesse.ligo.git_path()}/analysis/O4/LLO/drmi')
from aligo_newbs import ALIGOBeamSplitterFactory
#%%

# Open an image file
# Image(filename='/Users/raeddiab/Documents/PhD/Research/Finesse/FINESSE 3/ASC study/finesse-ligo/LHO/plots/ASC_REFL_HAM1_HAM2.JPEG')  

# Image(filename='/Users/raeddiab/Documents/PhD/Research/Finesse/FINESSE 3/ASC study/finesse-ligo/LHO/plots/aLIGOSeismicIsolation_LVEA_gIM2_p1_i20011-v8.JPEG')  



parentrepo = f'{finesse.ligo.git_path()}/' #str(finesse.ligo.git_path())
repopath = f'{finesse.ligo.git_path()}/analysis/O4/LHO_Thermalization/'

baffle=True            

if baffle== True:
    factory = ALIGOBeamSplitterFactory(f"{parentrepo}/LHO/yaml/lho_O4.yaml")
else:
    factory = ALIGOFactory(f"{parentrepo}/LHO/yaml/lho_O4.yaml")

factory.reset()
factory.options.QUAD_suspension_lho = QUADSuspension
factory.params.INPUT.LASER.power = 2 
factory.options.INPUT.add_IMC_and_IM1 = False
factory.options.LSC.add_output_detectors = True
factory.options.ASC.add = False
factory.options.thermal.add = True
factory.options.LSC.add_locks = True
factory.options.apertures.add = True       
factory.options.apertures.test_mass = True
factory.options.apertures.PR3_SR3 = True
factory.options.apertures.use_surface_profile = True
factory.options.apertures.BS_ITM=False  

factory.update_parameters(f"{parentrepo}/LHO/yaml/lho_addRH.yaml")
factory.update_parameters(f'{finesse.ligo.git_path()}/analysis/O4/LHO_Thermalization/data/run_cold_state/cold_state_yaml_solutions/cold_state_sol_0.yaml')

# model=factory.make()
# print(model.P_RH_ETMX)
#%%



scale=2

P_RH_ITMX = scale*factory.params.P_RH_ITMX * factory.params.RH_eff['ITMX']
P_RH_ITMY = scale*factory.params.P_RH_ITMY * factory.params.RH_eff['ITMY']
P_RH_ETMX = scale*factory.params.P_RH_ETMX * factory.params.RH_eff['ETMX']
P_RH_ETMY = scale*factory.params.P_RH_ETMY * factory.params.RH_eff['ETMY']

def add_detectors(lho):
    lho.add(fc.Beamsplitter('POP_BS', R=0.5, T=0.5))
    lho.connect(lho.PR2.p3, lho.POP_BS.p1)
    lho.add(fc.ReadoutDC('ReadoutDC_ASC_REFL_A', optical_node=lho.WFS_REFL_BS.p3.o,output_detectors=True))
    lho.add(fc.ReadoutDC('ReadoutDC_ASC_REFL_B', optical_node=lho.WFS_REFL_BS.p2.o,output_detectors=True))
    lho.add(fc.ReadoutDC('ReadoutDC_LSC_REFL_A', optical_node=lho.LSC_REFL_BS.p3.o,output_detectors=True))
    lho.add(fc.ReadoutDC('ReadoutDC_LSC_REFL_B', optical_node=lho.LSC_REFL_BS.p2.o,output_detectors=True))
    lho.add(fc.ReadoutDC('ReadoutDC_AS_C_QPD', optical_node=lho.AS_C.p1.i,output_detectors=True))
    lho.add(fc.ReadoutDC('ReadoutDC_POP_A', optical_node=lho.POP_BS.p3.o,output_detectors=True))
    lho.add(fc.ReadoutDC('ReadoutDC_POP_B', optical_node=lho.POP_BS.p2.o,output_detectors=True))

    lho.add(fc.ReadoutDC('ReadoutDC_ASC_AS_A', optical_node=lho.AS_A.p1.i,output_detectors=True))
    lho.add(fc.ReadoutDC('ReadoutDC_ASC_AS_B', optical_node=lho.AS_B.p1.i,output_detectors=True))
    return lho


# print(model.trace_forest)
def find_dep(node_name, tmodel):
    tf = tmodel.trace_forest
    node = tmodel.get(node_name)
    return tf.find_dependency_from_node(node).name


def mismatch_calculator(q1,q2,percentage=True):
    MM=(np.abs(q1 -q2))**2/(np.abs(q1-np.conjugate(q2)))**2
    if percentage==True:
        return 100*MM
    else:
        return MM


#%%

###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
use_real_data = True # needs to be true to plot data
maxx = 6000
import warnings
import pickle

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
initial_data_time =1418514918 # to do: add ability to specify time and pull data.
parentrepo=f'{finesse.ligo.git_path()}/'
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
P_pop = d[initial_data_time]['data']['H1:LSC-POP_A_LF_OUT16'].value[irange] / (1e6*0.054) #this is from microwatts to watts and there is 95% beam dumping
P_PRG = d[initial_data_time]['data']['H1:LSC-PR_GAIN_OUT16'].value[irange] 
Kappa_C = d[initial_data_time]['data']['H1:CAL-CS_TDEP_KAPPA_C_OUTPUT'].value[irange] # normalized to 1
RF18 = d[initial_data_time]['data']['H1:LSC-POPAIR_B_RF18_I_MON'].value[irange]*1.17/P_in #* 64.7/2700 W/cts per input W
RF90 = d[initial_data_time]['data']['H1:LSC-POPAIR_B_RF90_I_MON'].value[irange]*0.7/P_in #* 27.8/700  W/cts per input W
# # To do, could change RF18 and RF90 to Magnitude =  SQRT(I^2 + Q^2)

AGX = d[initial_data_time]['data']['H1:ASC-X_PWR_CIRC_OUT16'].value[irange]*2 /(P_in*P_PRG) #* 0.018 / P_in # 0.018 W/cts per input W
AGY = d[initial_data_time]['data']['H1:ASC-Y_PWR_CIRC_OUT16'].value[irange]*2 /(P_in*P_PRG) #* 0.018 / P_in # 0.018 W/cts per input W
# # To do, could change RF18 and RF90 to Magnitude =  SQRT(I^2 + Q^2)

###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
# %%
Powers={'Power_values':[2,10,25,50,60],'ASC_REFL_A':[],'ASC_REFL_B':[],'LSC_REFL_A':[],'LSC_REFL_B':[],
'AS_C_QPD':[],'ASC_AS_A':[],'POP_A':[],'POP_B':[],'ASC_AS_B':[],'PRG':[],'PRG9':[],'PRG45':[],
'PreflPRM':[],'Prefl':[],'Ppop':[],'Pas_c':[],'Pas':[],'Pin':[]}

Powers['Power_values'] = []#2
n_first = 3
n_rest =  0

threshold = 3300 #17000
first_segment = P_in[:threshold]
rest_segment = P_in[threshold:]
indices=[]#1800
for i in range(n_first):
    fraction = (i + 1) / n_first
    index_needed = int(len(first_segment) * fraction) - 1
    indices.append(index_needed)
    Powers['Power_values'].append(first_segment[index_needed])

for i in range(n_rest):
    fraction = (i + 1) / n_rest
    index_needed = int(len(rest_segment) * fraction) - 1
    indices.append(index_needed)
    Powers['Power_values'].append(rest_segment[index_needed])


################################################
for idx,P in enumerate(Powers['Power_values']):
    print(f"Working on {P}W")
    # factory.reset()
    lho = factory.make()
    lho=add_detectors(lho)
    lho.L0.P=P

    lho.modes(maxtem=7)
    lho.alpha_ITMX = factory.params.absorption.ITMX
    lho.alpha_ITMY = factory.params.absorption.ITMY
    lho.alpha_ETMX = factory.params.absorption.ETMX
    lho.alpha_ETMY = factory.params.absorption.ETMY
    lho.IRH_sub=factory.params.IRH_sub
    lho.IRH_srf=factory.params.IRH_srf
    lho.ERH_srf=factory.params.ERH_srf
    lho.ERH_sub=factory.params.ERH_sub

    lho.remove(lho.gIM2_p1_i)
    lho.add(
        finesse.components.Gauss(
            'gIM2_p1_i',
            lho.PRMAR.p2.i,
            w0x=factory.params.beam_waist.w0x,
            zx=factory.params.beam_waist.zx,
            w0y=factory.params.beam_waist.w0y,
            zy=factory.params.beam_waist.zy))


    if baffle==True:
    ################### BS baffles ############################
        factory.add_BS_baffles(lho, input_offset=0)
        lho.phase_level = 2 
        lho._settings.phase_config.v2_transmission_phase = False
        lho.beam_trace()
        # Reajusting loss in arms:
        # lho.modes( maxtem=4)
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
    else:
        lho.add(det.PowerDetector("M6_power_input",node=lho.M6.p1.i))
        lho.add(det.PowerDetector("M6_power_onput2",node=lho.M6.p2.o))
        lho.add(det.PowerDetector("M6_power_onput3",node=lho.M6.p3.o))
        lho.add(det.PowerDetector("LSC_REFL_BS_power",node=lho.LSC_REFL_BS.p1.i))
        lho.add(det.PowerDetector("LSC_REFL_BS_power2",node=lho.LSC_REFL_BS.p2.o))
        lho.add(det.PowerDetector("LSC_REFL_BS_power3",node=lho.LSC_REFL_BS.p3.o))
        lho.add(det.PowerDetector("Power_PRMARi",node=lho.PRMAR.p2.i))
        lho.add(det.PowerDetector("Power_PRMARo",node=lho.PRMAR.p2.o))

    def set_ringheaters(arm, P_RH_ITM, P_RH_ETM):
        lens = lho.get(f"ITM{arm}lens")
        lens.f = 1 / (1 / lens.f + P_RH_ITM * factory.params.IRH_sub)
        itm = lho.get(f"ITM{arm}")
        etm = lho.get(f"ETM{arm}")
        itm.Rc = 2 / (2 / itm.Rc + P_RH_ITM * factory.params.IRH_srf)
        etm.Rc = 2 / (2 / etm.Rc + P_RH_ETM * factory.params.ERH_srf)
        # print(factory.params.IRH_sub)
        # print(f"New values for ITM{arm} lens f: {lens.f.eval()}, ITM{arm} Rc: {itm.Rcx.eval()}, ETM{arm} Rc: {etm.Rcx.eval()}")


    locking_drag=False                 
    if locking_drag==True:
    #################### Locking drag ############################
        initial_params = {p.full_name: p.value for p in lho.all_parameters}
        lho_static_rh = lho.deepcopy()
        N_drag = 100  # Number of dragging steps
        gammas = np.linspace(0, 1, N_drag)

        # Iterate through the dragging process
        for _gamm in tqdm.tqdm(gammas):
            # Update ETMs (existing logic)
            for ETM, P_ERH in zip((lho_static_rh.ETMX, lho_static_rh.ETMY), (P_RH_ETMX, P_RH_ETMY)):
                ETM.Rc = 2 / (2 / float(initial_params[ETM.Rcx.full_name]) + 
                            _gamm * P_ERH * lho_static_rh.ERH_srf.eval())
            
            # Update ITMs and ITM lenses
            for ITMlens, ITM, P_RH_TM, bs_f in zip(
                (lho_static_rh.ITMXlens, lho_static_rh.ITMYlens),
                (lho_static_rh.ITMX, lho_static_rh.ITMY),
                (P_RH_ITMX, P_RH_ITMY),
                (225e3, -247e3),):
                # Update focal length using the dragging factor
                ITMlens.f.value = 1 / (1 / float(initial_params[ITMlens.f.full_name]) + 
                                    _gamm * P_RH_TM * lho_static_rh.IRH_sub.eval())
            
                # Update ITM radius of curvature (existing logic)
                ITM.Rc = 2 / (2 / float(initial_params[ITM.Rcx.full_name]) + 
                            _gamm * P_RH_TM * lho_static_rh.IRH_srf.eval())
            old_f = ITMlens.f.value
            ITMlens.f.value = 1 / (1 / old_f + 1 / bs_f)
                
            lho_static_rh.beam_trace()
            sol=lho_static_rh.run(
                fac.Series(
                fac.SetLockGains(gain_scale=0.5),
                    fac.RunLocks(
                        max_iterations=500,
                        exception_on_fail=False,
                    display_progress=False,
                    show_progress_bar=True,
                    ),
                fac.Noxaxis(name="noxaxis"),
                )
            )
    else:
        #### The effect of RH
        set_ringheaters("X", P_RH_ITMX, P_RH_ETMX)
        set_ringheaters("Y", P_RH_ITMY, P_RH_ETMY)
                # # #### This is some other correction I'm not sure about
        lho.ITMXlens.f = 1 / (1 / lho.ITMXlens.f + 1/(225e3)) 
        lho.ITMYlens.f = 1 / (1 / lho.ITMYlens.f + 1/(-247e3)) 

        print("lho.ITMXlens.f", 1/lho.ITMXlens.f.eval())
        print("lho.ITMYlens.f", 1/lho.ITMYlens.f.eval())
        if P > 3:
            sol = lho.run(fac.Series(InitialLockLIGO(gain_scale=0.4*(Powers['Power_values'][idx-1]/Powers['Power_values'][idx])), fac.SetLockGains(gain_scale=0.4),fac.Noxaxis())) #add a variable gain in InitialLockLIGO as the power changes
        elif P> 51:
            sol = lho.run(fac.Series(InitialLockLIGO(gain_scale=1*(Powers['Power_values'][idx-1]/Powers['Power_values'][idx])), fac.SetLockGains(gain_scale=0.4),fac.Noxaxis())) #add a variable gain in InitialLockLIGO as the power changes
        else:
            sol = lho.run(fac.Series(InitialLockLIGO(), fac.SetLockGains(gain_scale=0.4),fac.Noxaxis())) #add a variable gain in InitialLockLIGO as the power changes

    print("******************* AFTER LOCKING *******************")

    # print("The power at BS.p1.i", sol[-1]['P_test'])
    # print("The power at BS.p3.o", sol[-1]['P_test2'])


    # print(prop.total_acc_gouy)
    print("At IM2.p1.i",lho.IM2.p1.i.qx)
    print("At PRMAR.p1.i",lho.PRMAR.p2.i.qx)
    print("At BS.p1.i", lho.BS.p1.i.qx)
    print("At ITMX.p2.i", lho.ITMX.p2.i.qx)

    print("At ITMX.p1.i", lho.ITMX.p1.i.qx)
    print("At ITMX.p1.o", lho.ITMX.p1.o.qx)

    print("The gouy phase inside PRX is", lho.cavPRX.gouy)
    print("The gouy phase inside SRX is", lho.cavSRX.gouy)
    print("PR2.Rc", lho.PR2.Rc)
    print("PR3.Rc", lho.PR3.Rc)
    print("PRM.Rc", lho.PRM.Rc)

    # print("lho.gIM2_p1_i.qx.w0", lho.gIM2_p1_i.qx.w0)
    # print("lho.gIM2_p1_i.qy.w0", lho.gIM2_p1_i.qy.w0)
    print("lho.ITMX.Rc",lho.ITMX.Rc[0].value)
    print("lho.ITMY.Rc",lho.ITMY.Rc[0].value)
    print("lho.PRM.T",lho.PRM.T)
    print("absorption_ETMX",lho.alpha_ETMX)
    print("absorption_ITMX",lho.alpha_ITMX)
    print("absorption_ETMY",lho.alpha_ETMY)
    print("absorption_ITMY",lho.alpha_ITMY)
    print("length_PRM_PR2",lho.lp1.L)
    print("length_PR2_PR3",lho.lp2.L)
    print("length_PR3_BS",lho.lp3.L)
    # print("qx_z_init ", lho.gIM2_p1_i.qx.z)
    # print("qy_z_init ", lho.gIM2_p1_i.qy.z)

    print("SR2.Rc ", lho.SR2.Rc)
    print("SR3.Rc ", lho.SR3.Rc)
    print("SRM.Rc", lho.SRM.Rc)

    print("length_SRM_SR2",lho.ls1.L)
    print("length_SR2_SR3",lho.ls2.L)
    print("length_SR3_BS",lho.ls3.L)
    print("IRH_sub",lho.IRH_sub,
    "IRH_srf",lho.IRH_srf,
    "ERH_srf",lho.ERH_srf,
    "ERH_sub",lho.ERH_sub)
    print("PRG is",sol[-1]['PRG'],"PRG9 is",sol[-1]['PRG9'],"PRG45 is",sol[-1]['PRG45'])

    print("P_refl",sol[-1]['Prefl'])
    print("Ppop", sol[-1]['Ppop'])



    Powers['ASC_REFL_A'].append(sol[-1]['ReadoutDC_ASC_REFL_A_DC'])
    Powers['ASC_REFL_B'].append(sol[-1]['ReadoutDC_ASC_REFL_B_DC'])
    Powers['LSC_REFL_A'].append(sol[-1]['ReadoutDC_LSC_REFL_A_DC'])
    Powers['LSC_REFL_B'].append(sol[-1]['ReadoutDC_LSC_REFL_B_DC'])
    Powers['AS_C_QPD'].append(sol[-1]['ReadoutDC_AS_C_QPD_DC'])
    Powers['POP_A'].append(sol[-1]['ReadoutDC_POP_A_DC'])
    Powers['POP_B'].append(sol[-1]['ReadoutDC_POP_B_DC'])
    Powers['ASC_AS_A'].append(sol[-1]['ReadoutDC_ASC_AS_A_DC'])
    Powers['ASC_AS_B'].append(sol[-1]['ReadoutDC_ASC_AS_B_DC'])
    Powers['PRG'].append(sol[-1]['PRG'])
    Powers['PRG9'].append(sol[-1]['PRG9'])
    Powers['PRG45'].append(sol[-1]['PRG45'])
    Powers['Pin'].append(sol[-1]['Pin'])
    Powers['PreflPRM'].append(sol[-1]['PreflPRM'])
    Powers['Prefl'].append(sol[-1]['Prefl'])
    Powers['Ppop'].append(sol[-1]['Ppop'])
    Powers['Pas_c'].append(sol[-1]['Pas_c'])
    Powers['Pas'].append(sol[-1]['Pas'])
    print(find_dep("ITMX.p1.o", lho))
    print(find_dep("PRM.p2.i", lho))
    q1=lho.ITMX.p1.i.qx.q
    q2=lho.ITMX.p2.o.qx.q
    print("the mismatch at ITMX is" ,mismatch_calculator(q1,q2,percentage=True))
    # print("XARM MM manually calculated is ", 100*(np.abs(prop.q(lho.ITMX.p1.i).q -prop.q(lho.ITMX.p1.o).q ))**2/(np.abs(prop.q(lho.ITMXAR.p1.i).q-np.conjugate(prop.q(lho.ITMXAR.p1.o).q)))**2,'%')

# %%

from matplotlib.backends.backend_pdf import PdfPages

def plot_data(t_data, indices, Powers, P_in, P_refl, P_pop, P_PRG, RF18, RF90, save_pdf=False, plotloc='cold_state.pdf'):
    if save_pdf:
        pdf = PdfPages(plotloc)
    else:
        pdf = None

    def create_plot(x, y, indices_x, indices_y, y_label, title, powers_label, color='b', point_color='r'):
        # Create a figure and primary axis
        fig, ax1 = plt.subplots()
        ax1.plot(x, y, f'{color}-', label=y_label)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel(y_label, color=color)

        ax2 = ax1.twiny()

        ax2.plot(indices_x, indices_y, f'{point_color}o', markersize=5, label=powers_label)
        ax2.set_xlabel('Time [s]')
        ax2.spines['top'].set_color(point_color)
        ax2.tick_params(axis='x', labelcolor=point_color)
        ax2.set_xlim([0,6000])
        fig.legend(loc='upper left')

        plt.title(title)
        plt.tight_layout()

        if save_pdf:
            pdf.savefig()
            plt.close()
        else:
            plt.show()

    # Create and save/show each plot
    create_plot(t_data, P_in, t_data[indices], Powers['Pin'], 'Power (P_in)', 'Input power', 'Powers')
    create_plot(t_data, P_refl, t_data[indices], Powers['Prefl'], 'Prefl', 'Prefl', 'Powers')
    create_plot(t_data, P_pop, t_data[indices], Powers['Ppop'], 'Ppop', 'P_pop', 'Powers')
    create_plot(t_data, P_PRG, t_data[indices], Powers['PRG'], 'PRG', 'PRG', 'Powers')
    create_plot(t_data, RF18, t_data[indices], Powers['PRG9'], 'PRG9', 'PRG9', 'Powers')
    create_plot(t_data, RF90, t_data[indices], Powers['PRG45'], 'PRG45', 'PRG45', 'Powers')

    # Close the PDF if saving
    if save_pdf:
        pdf.close()



plotloc = '/Users/raeddiab/Documents/PhD/Research/Finesse/FINESSE 3/ASC study/finesse-ligo/LHO/plots/With_aperture_with_baffle.pdf'
# plot_data(t_data, indices, Powers, P_in, P_refl, P_pop, P_PRG, RF18, RF90, save_pdf=True, plotloc=plotloc)  # Save to PDF
plot_data(t_data, indices, Powers, P_in, P_refl, P_pop, P_PRG, RF18, RF90, save_pdf=False)  # Just show the plots
# print(plotloc)

#%%

#%%
%cd /Users/raeddiab/ligo-commissioning-modeling

from scipy.integrate import *
import numpy as np
from pyswarm import pso

PR2_Rc = 1
PR3_Rc = 1
PRM_Rc = 1
SRM_Rc = 1
SR2_Rc = 1
SR3_Rc = 1


qx_w0 = 1
qy_w0 = 1
absorption_ITMX=1
absorption_ITMY=1
absorption_ETMX=1
absorption_ETMY=1
ITMX_Rc=1
ITMY_Rc=1
trans_PRM=1
X_arm_loss=1
Y_arm_loss=1

length_PRM_PR2=1
length_PR2_PR3=1
length_PR3_BS=1
length_SRM_SR2=1
length_SR2_SR3=1
length_SR3_BS=1

qx_z = 1
qy_z = 1


IRH_sub=1
IRH_srf=1
ERH_srf=1
ERH_sub=1

particle_positions = []

def denormalize_params(params_norm, lb, ub):
    """Denormalizes parameters from [-1, 1] back to original range."""
    params_norm = np.array(params_norm)
    lb = np.array(lb)
    ub = np.array(ub)
    params = lb + (params_norm + 1) / 2 * (ub - lb)
    return params

def objective(params_norm):
    import finesse
    from finesse.ligo.factory import ALIGOFactory
    from finesse.ligo.actions import InitialLockLIGO, DARM_RF_to_DC
    from finesse.plotting import bode
    import finesse.analysis.actions as fac
    import finesse.analysis.actions as fa
    from finesse.ligo.factory import aligo
    import matplotlib.pylab as plt
    import sympy as sym
    from scipy.integrate import quad
    from finesse.ligo.ASC_controllers import get_controller
    from finesse.ligo.suspension import QUADSuspension
    import scipy
    import finesse.components as fc
    from IPython.display import Image
    import finesse.detectors as det
    from finesse.analysis.actions import FrequencyResponse
    import cmath
    from finesse.analysis.actions import (
        SensingMatrixDC,
        OptimiseRFReadoutPhaseDC)
    from sympy import integrate
    from finesse.analysis.actions.axes import Noxaxis
    finesse.init_plotting()
    from control import tf, root_locus
    import sys
    import matplotlib as mpl
    from finesse.symbols import Constant
    import pandas as pd 
    from IPython.display import Image
    import gpstime, pickle, glob, os, sys
    import yaml 
    import tqdm
    import sys 
    sys.path.append(f'{finesse.ligo.git_path()}/analysis/O4/LLO/drmi')
    from aligo_newbs import ALIGOBeamSplitterFactory



    # Open an image file
    # Image(filename='/Users/raeddiab/Documents/PhD/Research/Finesse/FINESSE 3/ASC study/finesse-ligo/LHO/plots/ASC_REFL_HAM1_HAM2.JPEG')  

    # Image(filename='/Users/raeddiab/Documents/PhD/Research/Finesse/FINESSE 3/ASC study/finesse-ligo/LHO/plots/aLIGOSeismicIsolation_LVEA_gIM2_p1_i20011-v8.JPEG')  

    def find_dep(node_name, tmodel):
        tf = tmodel.trace_forest
        node = tmodel.get(node_name)
        return tf.find_dependency_from_node(node).name


    def mismatch_calculator(q1,q2,percentage=True):
        MM=(np.abs(q1 -q2))**2/(np.abs(q1-np.conjugate(q2)))**2
        if percentage==True:
            return 100*MM
        else:
            return MM

    parentrepo = f'{finesse.ligo.git_path()}/' #str(finesse.ligo.git_path())
    repopath = f'{finesse.ligo.git_path()}/analysis/O4/GA_State_Optimization/'

    baffle=True           

    if baffle== True:
        factory = ALIGOBeamSplitterFactory(f"{parentrepo}/LHO/yaml/lho_O4.yaml")
    else:
        factory = ALIGOFactory(f"{parentrepo}/LHO/yaml/lho_O4.yaml")
    factory.update_parameters(f"{parentrepo}/LHO/yaml/lho_addRH.yaml")
    # factory.update_parameters(f'{finesse.ligo.git_path()}/analysis/O4/LHO_Thermalization/data/run_cold_state/cold_state_yaml_solutions/cold_state_sol_0.yaml')

    # factory.reset()
    factory.options.QUAD_suspension_lho = QUADSuspension
    factory.params.INPUT.LASER.power = 2 
    factory.options.INPUT.add_IMC_and_IM1 = False
    factory.options.LSC.add_output_detectors = True
    factory.options.ASC.add = False
    factory.options.thermal.add = True
    factory.options.LSC.add_locks = True
    factory.options.apertures.add = True       
    factory.options.apertures.test_mass = True
    factory.options.apertures.PR3_SR3 = True
    factory.options.apertures.use_surface_profile = True
    factory.options.apertures.BS_ITM=False  
    scale=2
    # RH_efficiency=factory.params.RH_eff
    RH_efficiency=1
    P_RH_ITMX = scale*factory.params.P_RH_ITMX * RH_efficiency#['ITMX']/RH_efficiency['ITMX']
    P_RH_ITMY = scale*factory.params.P_RH_ITMY * RH_efficiency#['ITMY']/RH_efficiency['ITMY']
    P_RH_ETMX = scale*factory.params.P_RH_ETMX * RH_efficiency#['ETMX']/RH_efficiency['ETMX']
    P_RH_ETMY = scale*factory.params.P_RH_ETMY * RH_efficiency#['ETMY']/RH_efficiency['ETMY']


    import corner

    from scipy.optimize import differential_evolution
    import numpy as np

    try:
        params = denormalize_params(params_norm, lb, ub)
        (
            PRM_Rc,PR2_Rc, PR3_Rc, qx_w0, qy_w0, absorption_ITMX, absorption_ITMY,
            absorption_ETMX, absorption_ETMY, ITMX_Rc,ITMY_Rc,
            trans_PRM, X_arm_loss,Y_arm_loss,
            length_PRM_PR2,length_PR2_PR3,length_PR3_BS,SRM_Rc,SR2_Rc, SR3_Rc,
            length_SRM_SR2,length_SR2_SR3, length_SR3_BS,qx_z, qy_z,IRH_sub,IRH_srf,ERH_srf,ERH_sub
        ) = params 


    
        # Initialize LHO simulation
        lho = factory.make()
        lho.remove(lho.gIM2_p1_i)
        lho.add(fc.gauss.Gauss("gIM2_p1_i",node=lho.PRMAR.p2.i,w0x=1.015e-3, zx=6,w0y=1.015e-3, zy=6))
        # Update simulation parameters
        lho.PR2.Rc = PR2_Rc
        lho.PR3.Rc = PR3_Rc
        lho.PRM.Rc = PRM_Rc
        lho.gIM2_p1_i.qx.w0 = qx_w0
        lho.gIM2_p1_i.qy.w0 = qy_w0
        lho.gIM2_p1_i.qx.z = qx_z
        lho.gIM2_p1_i.qy.z = qy_z

        lho.X_arm_loss = X_arm_loss
        lho.Y_arm_loss = Y_arm_loss
        lho.alpha_ITMX = absorption_ITMX
        lho.alpha_ITMY = absorption_ITMY
        lho.alpha_ETMX = absorption_ETMX
        lho.alpha_ETMY = absorption_ETMY
        lho.lp1.L = length_PRM_PR2
        lho.lp2.L = length_PR2_PR3
        lho.lp3.L = length_PR3_BS

        lho.SR2.Rc = SR2_Rc
        lho.SR3.Rc = SR3_Rc
        lho.SRM.Rc = SRM_Rc
        lho.ls1.L = length_SRM_SR2
        lho.ls2.L = length_SR2_SR3
        lho.ls3.L = length_SR3_BS
        # lho.ITMX.Rc = ITMX_Rc
        # lho.ITMY.Rc = ITMY_Rc  # Assuming symmetry
        lho.PRM.T = trans_PRM

        lho.IRH_sub=IRH_sub
        lho.IRH_srf=IRH_srf
        lho.ERH_srf=ERH_srf
        lho.ERH_sub=ERH_sub

        lho.L0.P = 1.981
    

        # print(find_dep("ITMX.p1.o", lho))
        # print(find_dep("PRM.p2.i", lho))
        q1=lho.ITMX.p1.i.qx.q
        q2=lho.ITMX.p2.o.qx.q
        MM=mismatch_calculator(q1,q2,percentage=True)
        # print("the mismatch at ITMX is" ,MM)
        lho.modes(maxtem=4)

        # Add power detectors
        lho.add(det.PowerDetector("P_test", node=lho.BS.p1.i))
        lho.add(det.PowerDetector("P_test2", node=lho.BS.p3.o))
        print("XARM loss", lho.X_arm_loss)
        print("YARM loss", lho.Y_arm_loss)
        print("lho.ITMX.Rc",lho.ITMX.Rc[0])
        print("lho.ITMY.Rc",lho.ITMY.Rc[0])
        # Handle baffles if enabled
        if baffle:
            factory.add_BS_baffles(lho, input_offset=0)
            lho.phase_level = 2
            lho._settings.phase_config.v2_transmission_phase = False
            lho.beam_trace()

            eigx = lho.run("eigenmodes(cavXARM, 0)")
            eigy = lho.run("eigenmodes(cavYARM, 0)")

            lho.X_arm_loss -= eigx.loss(True)[1][0]
            lho.Y_arm_loss -= eigy.loss(True)[1][0]
        else:
            lho.add(det.PowerDetector("M6_power_input", node=lho.M6.p1.i))
            lho.add(det.PowerDetector("M6_power_onput2", node=lho.M6.p2.o))
            lho.add(det.PowerDetector("M6_power_onput3", node=lho.M6.p3.o))
            lho.add(det.PowerDetector("LSC_REFL_BS_power", node=lho.LSC_REFL_BS.p1.i))

        # Define ring heater adjustments
        def set_ringheaters(arm, P_RH_ITM, P_RH_ETM):
            lens = lho.get(f"ITM{arm}lens")
            lens.f = 1 / (1 / lens.f + P_RH_ITM * lho.IRH_sub)
            itm = lho.get(f"ITM{arm}")
            etm = lho.get(f"ETM{arm}")
            itm.Rc = 2 / (2 / itm.Rc + P_RH_ITM * lho.IRH_srf)
            etm.Rc = 2 / (2 / etm.Rc + P_RH_ETM * lho.ERH_srf)
            # print("IRH is", lho.IRH_srf,"ERH is", ERH_srf)

        # Apply ring heater corrections
        set_ringheaters("X", P_RH_ITMX, P_RH_ETMX)
        set_ringheaters("Y", P_RH_ITMY, P_RH_ETMY)
        # #### This is some other correction I'm not sure about
        lho.ITMXlens.f = 1 / (1 / lho.ITMXlens.f + 1/(225e3)) 
        lho.ITMYlens.f = 1 / (1 / lho.ITMYlens.f + 1/(-247e3)) 

        # Run the simulation and extract outputs
        sol = lho.run(fac.Series(InitialLockLIGO(), DARM_RF_to_DC()))
        PRG, PRG9, PRG45 = sol[-1]["PRG"], sol[-1]["PRG9"], sol[-1]["PRG45"]
        PRX_gouy, SRX_gouy = (lho.cavPRX.gouy[0]+lho.cavPRX.gouy[1])/2,( lho.cavSRX.gouy[0]+lho.cavSRX.gouy[1])/2
        P_refl=sol[-1]['Prefl']
        P_pop=sol[-1]['Ppop']
        
        print("The gouy phase inside PRX is", lho.cavPRX.gouy)
        print("The gouy phase inside SRX is", lho.cavSRX.gouy)
        print("PRM.Rc", lho.PRM.Rc)
        print("PR2.Rc", lho.PR2.Rc)
        print("PR3.Rc", lho.PR3.Rc)
        print("lho.gIM2_p1_i.qx.w0", lho.gIM2_p1_i.qx.w0)
        print("lho.gIM2_p1_i.qy.w0", lho.gIM2_p1_i.qy.w0)
        # print("lho.ITMX.Rc",lho.ITMX.Rc)
        # print("lho.ITMY.Rc",lho.ITMY.Rc)
        print("lho.PRM.T",lho.PRM.T)
        print("absorption_ETMX",absorption_ETMX)
        print("absorption_ITMX",absorption_ITMX)
        print("absorption_ETMY",absorption_ETMY)
        print("absorption_ITMY",absorption_ITMY)
        print("length_PRM_PR2",length_PRM_PR2)
        print("length_PR2_PR3",length_PR2_PR3)
        print("length_PR3_BS",length_PR3_BS)
        print("qx_z_init ", qx_z)
        print("qy_z_init ", qy_z)

        print("SRM.Rc ", lho.SRM.Rc)
        print("SR2.Rc ", lho.SR2.Rc)
        print("SR3.Rc ", lho.SR3.Rc)
        print("length_SRM_SR2",length_SRM_SR2)
        print("length_SR2_SR3",length_SR2_SR3)
        print("length_SR3_BS",length_SR3_BS)
        print("IRH_sub",lho.IRH_sub,
        "IRH_srf",lho.IRH_srf,
        "ERH_srf",lho.ERH_srf,
        "ERH_sub",lho.ERH_sub)

        print("PRG is",PRG,"PRG9 is",PRG9,"PRG45 is",PRG45)
        print("PRX gouy",PRX_gouy)
        print("SRX gouy",SRX_gouy)
        print("P_refl",P_refl)
        print("P_pop",P_pop)
        # print("The mismatch is", MM)


        # Define the target parameters, their corresponding acceptable ranges, and weights
        parameter_info = {
            "P_refl": {"lower": 0.001, "upper": 0.0024, "weight": 1},
            "P_pop": {"lower": 0.02, "upper": 0.026, "weight": 1},
            "PRG": {"lower": 53, "upper": 55, "weight": 10},
            "PRG9": {"lower": 85, "upper": 95, "weight": 1},
            "PRG45": {"lower": 9.5, "upper": 11.5, "weight": 1},
            "PRX_gouy": {"lower": 42, "upper": 46, "weight": 1},
            "SRX_gouy": {"lower": 37, "upper": 41, "weight": 1}
        }
        outputs = np.array([P_refl,P_pop,PRG,PRG9, PRG45, PRX_gouy, SRX_gouy]) # PRG9, PRG45,

        # Compute the penalty for out-of-bound values
        lower_penalty = np.maximum(0, np.array([parameter_info[param]["lower"] for param in parameter_info]) - outputs) ** 2
        upper_penalty = np.maximum(0, outputs - np.array([parameter_info[param]["upper"] for param in parameter_info])) ** 2

        # Extract the weights and calculate the weighted sum of penalties
        weights = np.array([parameter_info[param]["weight"] for param in parameter_info])
        
        error = np.sum(weights * (lower_penalty + upper_penalty))

        if np.isnan(error):
            print("NAAAAAAAAAAAAAAAAN")
        else:
            print("*************************************")

            print("Error:", np.sqrt(error))
            print("*************************************")
            particle_positions.append(params)

        return error

    except Exception as e:
        # Handle errors and return a large penalty
        print(f"Error during evaluation: {e}")
        return 1e6  # Large penalty for invalid parameter combinations

# Function to create valid bounds but allow negative values
def custom_bounds(value):
    # If value is positive, apply 98% to 102% range
    if value >= 0:
        return (0.9995 * value,1.0005  * value)
    # If value is negative, keep the negative range for flexibility
    else:
        # Allow negative values and scale them proportionally
        return (1.0005 * value, 0.9995 * value)  # Upper bound should be more negative than lower bound

# Define bounds with custom function, allowing negative values
bounds = [
    custom_bounds(PRM_Rc),  # PR2_Rc_x
    custom_bounds(PR2_Rc),  # PR2_Rc_x
    (PR3_Rc-0.17,PR3_Rc+0.17),  # PR3_Rc_x #there is a 17 cm uncertinity
    (qx_w0-0.0005,qx_w0+0.00050),  # qx_w0
    (qx_w0-0.0005,qx_w0+0.00050),  # qy_w0
    (0.1e-6,0.7e-6),  # absorption_ETMX
    (0.1e-6,0.7e-6),  # absorption_ETMX
    (0.1e-6,0.7e-6),  # absorption_ETMX
    (0.1e-6,0.7e-6),  # absorption_ETMX
    (1940.2,1940.4),  # ITMX.Rc
    (1940.1,1940.3),  # ITMY.Rc
    (0.027,0.0315),  # PRM.T 
    (40e-6,70e-6),  # X_arm_loss
    (40e-6,70e-6),  # Y_arm_loss
    (length_PRM_PR2-0.005,length_PRM_PR2+0.005), #length_PRM_PR2
    (length_PR2_PR3-0.005,length_PR2_PR3+0.005), #length_PR2_PR3
    (length_PR3_BS-0.005,length_PR3_BS+0.005), #length_PR3_BS
    custom_bounds(SRM_Rc),  # SR2
    custom_bounds(SR2_Rc),  # SR2
    (SR3_Rc-0.17,SR3_Rc+0.17),  # SR3.Rc
    (length_SRM_SR2-0.005,length_SRM_SR2+0.005), #length_SRM_SR2
    (length_SR2_SR3-0.005,length_SR2_SR3+0.005), #length_SR2_SR3
    (length_SR3_BS-0.005,length_SR3_BS+0.005), #length_SR3_BS
    (qx_z-0.5,qx_z+0.5),  # qx_z
    (qy_z-0.5,qy_z+0.5),  # qy_z 
    (-15e-6,-8e-6), #IRH_sub
    (0.8e-6,1.2e-6), #IRH_srf
    (0.8e-6,1.2e-6), #ERH_srf
    (-15e-6,-8e-6), #ERH_sub
]

def normalize_bounds(lb, ub):
    """Normalizes bounds to [-1, 1] using the provided lb and ub."""
    lb = np.array(lb)
    ub = np.array(ub)
    lb_norm = -1 + 2 * (lb - lb) / (ub - lb)  # Simplified: lb is min, ub is max
    ub_norm = -1 + 2 * (ub - lb) / (ub - lb)  # Simplified: ub is max, lb is min
    return lb_norm, ub_norm


# Extract lower and upper bounds from bounds list and convert to NumPy arrays
lb, ub = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])

lb_norm, ub_norm = normalize_bounds(lb, ub)

# Modify the pso call to pass lb and ub to the objective function
best_params, best_error = pso(
    objective,  # Pass lb and ub using a lambda function
    lb_norm, ub_norm,
    swarmsize=50, maxiter=200, minstep=0.1, minfunc=0.01,
    omega=0.7, phip=2.0, phig=2.0)

print(best,params)
#%%
# Create the corner plot
import corner
corner.corner(np.array(particle_positions), labels=[
    "PRM_Rc", "PR2_Rc", "PR3_Rc", "qx_w0", "qy_w0", "abs_ITMX", "abs_ITMY",
    "abs_ETMX", "abs_ETMY", "ITMX_Rc", "ITMY_Rc", "trans_PRM", "X_arm_loss", "Y_arm_loss",
    "length_PRM_PR2", "length_PR2_PR3", "length_PR3_BS", "SRM_Rc", "SR2_Rc", "SR3_Rc",
    "length_SRM_SR2", "length_SR2_SR3", "length_SR3_BS", "qx_z", "qy_z", "IRH_sub", "IRH_srf",
    "ERH_srf", "ERH_sub"
], show_titles=True)

# plt.savefig(f"/Users/raeddiab/Documents/PhD/Research/Finesse/FINESSE 3/ASC study/finesse-ligo/LHO/plots/corner_plot.pdf")
plt.show()
#%%



