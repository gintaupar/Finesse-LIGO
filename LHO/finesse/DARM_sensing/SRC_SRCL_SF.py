# The point of this file is to replicate the plots from 
# https://dcc.ligo.org/public/0006/T0900511/004/T0900511_v4.pdf
#I'll start with plot 8/
# %%
import numpy as np
import pytest
import finesse
import finesse.ligo
from finesse.ligo.factory import ALIGOFactory
from finesse.ligo.actions import InitialLockLIGO, DARM_RF_to_DC
from finesse.plotting import bode
import finesse.analysis.actions as fac
import finesse.analysis.actions as fa
from finesse.ligo.factory import aligo
import matplotlib.pylab as plt
import sympy as sym
from scipy.integrate import quad
from finesse.ligo.QUAD_control import get_locking_filters

from finesse.ligo.ASC_controllers import get_controller
from finesse.ligo.suspension import QUADSuspension
from scipy.integrate import *
import scipy
import finesse.components as fc
from IPython.display import Image
import finesse.detectors as det
from finesse.analysis.actions import FrequencyResponse
import cmath
import pandas as pd
from sympy import integrate
from finesse.analysis.actions.axes import Noxaxis
finesse.init_plotting()
# from control import tf, root_locus
from finesse.components.mechanical import Pendulum
import finesse_ligo
from functions import *
# finesse.plot_
#let's plot OLG TF
parentrepo = f'{finesse.ligo.git_path()}/' #str(finesse.ligo.git_path())

factory = aligo.ALIGOFactory(f"{parentrepo}/LHO/yaml/lho_O4.yaml")
# factory.reset()
# factory.options.QUAD_suspension_model = QUADSuspension
factory.options.QUAD_suspension_model = finesse_ligo.suspension.QUADSuspension
factory.options.ASC.add = True
factory.options.ASC.add_DOFs = True
factory.options.ASC.add_locks=True
factory.options.ASC.add_output_detectors=True
factory.options.ASC.add_readouts = True
factory.options.ASC.add_AC_loops = True
factory.options.ASC.close_AC_loops = True  


F_Hz=np.geomspace(7, 1000, 100)

#%%

model1=factory.make()
model1.modes(maxtem=3) #try modes("off")
model1.fsig.f=1
minimize_MM(model1)
detectors_on=True
if detectors_on==True:
    add_amplitude_detectors(model1)
cav_mismatches1=get_cav_mismatches(model1,print_tables=False)


SRM_tilt=np.linspace(-21e-6,20.0e-6,5)
colors = plt.cm.rainbow(np.linspace(0, 1, len(SRM_tilt)))
data_output={'HG00_amp':[],'HG10_amp':[],'HG20_amp':[],'HG01_amp':[],'HG02_amp':[],'SRCL_detuning':[],'gouy_shift':[]}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
for i, color in zip(SRM_tilt, colors):
    with model1.temporary_parameters():
        model1.SRM.ybeta = i
        out=model1.run(fac.Series(
        (aligo.InitialLock()),
        aligo.DARM_RF_to_DC(),
        fac.Noxaxis(),
        ))
        print(out[-1]['a_u45_00_src'])
        print(out[-1]['a_l45_00_src'])
        print(out[-1]['PRG45'])
        data_output['gouy_shift'].append(model1.SRM.p1.i.qy.gouy() * 180/np.pi )
        data_output['SRCL_detuning'].append(model1.SRCL.DC.value)
        if detectors_on==True:
            data_output['HG00_amp'].append(out['noxaxis']['HG00_amp'])
            data_output['HG10_amp'].append(out['noxaxis']['HG10_amp'])
            data_output['HG20_amp'].append(out['noxaxis']['HG20_amp'])
            data_output['HG01_amp'].append(out['noxaxis']['HG01_amp'])
            data_output['HG02_amp'].append(out['noxaxis']['HG02_amp'])
        sol =model1.run(fa.Series(fa.FrequencyResponse(F_Hz, model1.DARM.AC.i, model1.AS.DC.o, name="fresp")))
        ax1.loglog(F_Hz, np.abs(sol['AS.DC.o', 'DARM.AC.i']), color=color, label=f"SRM tilt {i*1e6:.1f} µrad")
        ax2.semilogx(F_Hz, np.angle(sol['AS.DC.o', 'DARM.AC.i'])*180/np.pi, color=color, label=f"SRM tilt {i*1e6:.1f} µrad")

ax1.set_xlabel("Frequency [Hz]")
ax1.set_ylabel("Magnitude [AS/DARM]")
ax1.grid(True)
ax1.legend()

ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Phase [deg]")
ax2.grid(True)
ax2.legend()

plt.suptitle("DARM sensing function")
plt.tight_layout()
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/DARM_SF_vs_SRM_tilt.pdf")    
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/DARM_SF_vs_SRM_tilt.png")
plt.show()
#%%
plt.plot(SRM_tilt,(np.array(data_output['SRCL_detuning'])),'o')
plt.xlabel("SRM tilt [urad]")
plt.ylabel("SRCL.DC")
plt.title("SRCL.DC vs SRM tilt")
plt.grid(True)
plt.tight_layout()
plt.show()

if detectors_on==True:
    plt.plot(SRM_tilt,np.angle(data_output['HG00_amp'])*180/np.pi,'o',label="HG00")
    # plt.plot(SRM_tilt,np.angle(data_output['HG10_amp'])*180/np.pi,'x',label="HG10")
    plt.plot(SRM_tilt,np.angle(data_output['HG20_amp'])*180/np.pi,'d',label="HG20")
    plt.plot(SRM_tilt,np.angle(data_output['HG01_amp'])*180/np.pi,'s',label="HG01")
    plt.plot(SRM_tilt,np.angle(data_output['HG02_amp'])*180/np.pi,'v',label="HG02")
    plt.xlabel("SRM tilt [urad]")
    plt.ylabel("Phase [deg]")
    plt.title("Phase of Carrier HG00, HG10, HG20, HG01, HG02")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plt.plot(SRM_tilt,data_output['gouy_shift'])
plt.xlabel("SRM tilt [urad]")
plt.ylabel("Gouy shift [deg]")
plt.title("Gouy shift vs SRM tilt")
plt.grid(True)
plt.tight_layout()
plt.show()
#%%



model2=factory.make()
model2.modes(maxtem=2) #try modes("off")
model2.fsig.f=1
minimize_MM(model2)
detectors_on=True
if detectors_on==True:
    add_amplitude_detectors(model2)
cav_mismatches1=get_cav_mismatches(model2,print_tables=False)
# model2.SRM.R=0.94

SR3_Rc=np.linspace(-0.001,0.0005,5)
colors = plt.cm.rainbow(np.linspace(0, 1, len(SR3_Rc)))
data_output={'HG00_amp':[],'HG10_amp':[],'HG20_amp':[],'HG01_amp':[],'HG02_amp':[],'SRCL_detuning':[],'gouy_shift':[],'SR3_Rc':[]
             ,'SRC_gouy':[],'a_u45_00_src':[],'a_l45_00_src':[],'PRG45':[]}

labels = {'graph_labels':[]}


data={'XYARM':[],'SRXXARM':[],'SRYYARM':[],'PRXXARM':[],'PRYYARM':[],'SRCL_DC':[],'ITMX_RoC_change':[],'ITMY_RoC_change':[],'XYARM_MM':[]
,'ITMX_w':[],'ITMY_w':[],'ETMX_w':[],'ETMY_w':[]}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
for i, color in zip(SR3_Rc, colors):
    with model2.temporary_parameters():
        model2.SR3.Rc *= (1+i)
        # print(model2.SR2.Rc)
        data_output['SR3_Rc'].append(model2.SR3.Rc[0])
        out=model2.run(fac.Series(
        aligo.InitialLock(),
        aligo.DARM_RF_to_DC(),
        fac.Noxaxis(),
        ))
        print(out[-1]['a_u45_00_src'])
        print(out[-1]['a_l45_00_src'])
        print(out[-1]['PRG45'])
        gouy=(model2.cavSRX.gouy+model2.cavSRY.gouy)/2
        data_output['SRC_gouy'].append(gouy)
        data_output['a_u45_00_src'].append(out[-1]['a_u45_00_src'])
        data_output['a_l45_00_src'].append(out[-1]['a_l45_00_src'])
        data_output['PRG45'].append(out[-1]['PRG45'])


        cav_mismatches2=get_cav_mismatches(model2,print_tables=False)


        label = f"SR3 RoC: {((2/model2.SR3.Rc)/1e-3)[0]:.2f} mD" #f"{(XYARMs_MM/1e-2):.2f} %"
        labels['graph_labels'].append(label)
        XYARMs_MM=(float(cav_mismatches2['XARM_YARM_x'])+float(cav_mismatches2['XARM_YARM_y']))/2
        SRXXARMs_MM=(float(cav_mismatches2['SRX_XARM_x'])+float(cav_mismatches2['SRX_XARM_y']))/2
        SRYYARMs_MM=(float(cav_mismatches2['SRY_YARM_x'])+float(cav_mismatches2['SRY_YARM_y']))/2

        PRXXARMs_MM=(float(cav_mismatches2['PRX_XARM_x'])+float(cav_mismatches2['PRX_XARM_y']))/2
        PRYYARMs_MM=(float(cav_mismatches2['PRY_YARM_x'])+float(cav_mismatches2['PRY_YARM_y']))/2

        data['PRXXARM'].append(PRXXARMs_MM)
        data['PRYYARM'].append(PRYYARMs_MM)

        data['SRXXARM'].append(SRXXARMs_MM)
        data['SRYYARM'].append(SRYYARMs_MM)
        data['XYARM'].append(XYARMs_MM)
        data['SRCL_DC'].append(model2.SRCL.DC.value)

        data['XYARM_MM'].append(XYARMs_MM)
        data['ITMX_w'].append(model2.ITMX.p1.i.qx.w)
        data['ITMY_w'].append(model2.ITMY.p1.i.qx.w)
        data['ETMX_w'].append(model2.ETMX.p1.i.qx.w)
        data['ETMY_w'].append(model2.ETMY.p1.i.qx.w)

        data_output['gouy_shift'].append(model2.SRM.p1.i.qy.gouy() * 180/np.pi )
        data_output['SRCL_detuning'].append(model2.SRCL.DC.value)
        if detectors_on==True:
            data_output['HG00_amp'].append(out['noxaxis']['HG00_amp'])
            data_output['HG10_amp'].append(out['noxaxis']['HG10_amp'])
            data_output['HG20_amp'].append(out['noxaxis']['HG20_amp'])
            data_output['HG01_amp'].append(out['noxaxis']['HG01_amp'])
            data_output['HG02_amp'].append(out['noxaxis']['HG02_amp'])        
        sol =model2.run(fa.Series(fa.FrequencyResponse(F_Hz, model2.DARM.AC.i, model2.AS.DC.o, name="fresp")))
        ax1.loglog(F_Hz, np.abs(sol['AS.DC.o', 'DARM.AC.i']), color=color, label=label)
        ax2.semilogx(F_Hz, np.angle(sol['AS.DC.o', 'DARM.AC.i'])*180/np.pi, color=color, label=label)

ax1.set_xlabel("Frequency [Hz]")
ax1.set_ylabel("Magnitude [AS/DARM]")
ax1.grid(True)
ax1.legend()

ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Phase [deg]")
ax2.grid(True)
ax2.legend()

plt.suptitle("DARM sensing function")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [AS/DARM]")
plt.title("DARM sensing function")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SR3_RoC_SF.pdf")    
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SR3_RoC_SF.png")
plt.show()

#%%
create_data_table(labels, data,name="SR3_RoC", save_md=True, save_png=True)

plt.plot(data_output['SR3_Rc'],(np.array(data_output['SRCL_detuning'])),color='green')
plt.xlabel("SR2 Rc [m]")
plt.ylabel("SRCL.DC [deg]")
plt.title("SRCL.DC vs SR2 Rc")
plt.grid(True)
plt.tight_layout()
plt.show()

if detectors_on==True:
    fig, ax1 = plt.subplots()

    # Plot HG00 on the left y-axis
    ax1.plot(data_output['SR3_Rc'], np.angle(data_output['HG00_amp']) * 180 / np.pi, label="HG00", color='C0')
    ax1.set_xlabel("SR2 Rc [m]")
    ax1.set_ylabel("HG00 Phase [deg]", color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax1.grid(True)

    # Create a second y-axis
    ax2 = ax1.twinx()

    # Plot HG20 and HG02 on the right y-axis
    ax2.plot(data_output['SR3_Rc'], np.angle(data_output['HG20_amp']) * 180 / np.pi, '.-', label="HG20", color='C1')
    ax2.plot(data_output['SR3_Rc'], np.angle(data_output['HG02_amp']) * 180 / np.pi, '.--', label="HG02", color='C2')
    ax2.set_ylabel("HG20 / HG02 Phase [deg]", color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title("Phase of Carrier HG Modes")
    plt.tight_layout()
    plt.show()



# plt.plot(data_output['SR3_Rc'],data_output['gouy_shift'],color='red')
# plt.xlabel("SR2 Rc [m]")
# plt.ylabel("Gouy shift [deg]")
# plt.title("Gouy shift vs SR2 Rc")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# %%
plt.plot(data_output['SR3_Rc'],np.array(data_output['SRC_gouy'])[:,0],color='blue',label='x')
plt.plot(data_output['SR3_Rc'],np.array(data_output['SRC_gouy'])[:,1],color='red',label='y')
plt.xlabel("SR2 Rc [m]")
plt.ylabel("SRC RT Gouy [deg]")
plt.title("SRC RT Gouy vs SR2 Rc")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# %%

plt.plot(data_output['SR3_Rc'],data_output['PRG45']/(np.abs(np.array(data_output['a_u45_00_src']))**2+np.abs(np.array(data_output['a_l45_00_src']))**2),color='blue',label='a_00_src')
plt.xlabel("SR2 Rc [m]")
plt.ylabel("SRC RT Gouy [deg]")
plt.title("SRC RT Gouy vs SR2 Rc")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# %%


model3=factory.make()
model3.modes(maxtem=2) #try modes("off")
model3.fsig.f=1
minimize_MM(model3)
detectors_on=True
if detectors_on==True:
    add_amplitude_detectors(model3)
cav_mismatches1=get_cav_mismatches(model3,print_tables=False)
# model3.SRM.R=0.94

CM2_Rc=np.linspace(-0.6,10,3)
colors = plt.cm.rainbow(np.linspace(0, 1, len(CM2_Rc)))
data_output={'HG00_amp':[],'HG10_amp':[],'HG20_amp':[],'HG01_amp':[],'HG02_amp':[],'SRCL_detuning':[],'gouy_shift':[],'CM2_Rc':[]
             ,'SRC_gouy':[],'a_u45_00_src':[],'a_l45_00_src':[],'PRG45':[]}
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
for i, color in zip(CM2_Rc, colors):
    with model3.temporary_parameters():
        model3.OMC_CM2.Rc *= (1+i)
        cav_mismatches1=get_cav_mismatches(model3,print_tables=False)
        # print(cav_mismatches1)
        # print(model3.OMC_CM2.Rc)
        data_output['CM2_Rc'].append(model3.OMC_CM2.Rc[0])
        out=model3.run(fac.Series(
        aligo.InitialLock(),
        aligo.DARM_RF_to_DC(),
        fac.Noxaxis(),
        ))
        # print(out[-1]['a_u45_00_src'])
        # print(out[-1]['a_l45_00_src'])
        # print(out[-1]['PRG45'])
        gouy=(model3.cavSRX.gouy+model3.cavSRY.gouy)/2
        data_output['SRC_gouy'].append(gouy)
        data_output['a_u45_00_src'].append(out[-1]['a_u45_00_src'])
        data_output['a_l45_00_src'].append(out[-1]['a_l45_00_src'])
        data_output['PRG45'].append(out[-1]['PRG45'])
        data_output['gouy_shift'].append(model3.SRM.p1.i.qy.gouy() * 180/np.pi )
        data_output['SRCL_detuning'].append(model3.SRCL.DC.value)
        if detectors_on==True:
            data_output['HG00_amp'].append(out['noxaxis']['HG00_amp'])
            data_output['HG10_amp'].append(out['noxaxis']['HG10_amp'])
            data_output['HG20_amp'].append(out['noxaxis']['HG20_amp'])
            data_output['HG01_amp'].append(out['noxaxis']['HG01_amp'])
            data_output['HG02_amp'].append(out['noxaxis']['HG02_amp'])        
        sol =model3.run(fa.Series(fa.FrequencyResponse(F_Hz, model3.DARM.AC.i, model3.AS.DC.o, name="fresp")))
        ax1.loglog(F_Hz, np.abs(sol['AS.DC.o', 'DARM.AC.i']), color=color, label=f"CM2 Rc {model3.OMC_CM2.Rc[0]:.3f} m")
        ax2.semilogx(F_Hz, np.angle(sol['AS.DC.o', 'DARM.AC.i'])*180/np.pi, color=color, label=f"CM2 Rc {model3.OMC_CM2.Rc[0]:.3f} m")

ax1.set_xlabel("Frequency [Hz]")
ax1.set_ylabel("Magnitude [AS/DARM]")
ax1.grid(True)
ax1.legend()

ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Phase [deg]")
ax2.grid(True)
ax2.legend()

plt.suptitle("DARM sensing function")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [AS/DARM]")
plt.title("DARM sensing function")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SR2_RoC_SF.pdf")    
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SR2_RoC_SF.png")
plt.show()

# %%
plt.plot(data_output['CM2_Rc'],(np.array(data_output['SRCL_detuning'])),color='green')
plt.xlabel("CM2 Rc [m]")
plt.ylabel("SRCL.DC [deg]")
plt.title("SRCL.DC vs CM2 Rc")
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
