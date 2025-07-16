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
model1.modes(maxtem=4) #try modes("off")
model1.fsig.f=1
# minimize_MM(model1)
# model1.ITMX.ybeta=1e-9
detectors_on=False
if detectors_on==True:
    add_amplitude_detectors(model1)
cav_mismatches1=get_cav_mismatches(model1,print_tables=False)

model1.run(fac.Series(
(aligo.InitialLock()),
aligo.DARM_RF_to_DC()
))
SRCL_DC_range=np.linspace(-0.005,0.005,5)
colors = plt.cm.rainbow(np.linspace(0, 1, len(SRCL_DC_range)))
data_output={'HG00_SB':[],'HG00_amp':[],'HG10_amp':[],'HG20_amp':[],'HG01_amp':[],'HG02_amp':[],'SRCL_detuning':[],'gouy_shift':[]}
for i, color in zip(SRCL_DC_range, colors):
    with model1.temporary_parameters():
        model1.SRCL.DC *= (1+i)
        out=model1.run()
        # print(model1.cavSRX.round_trip_optical_length)
        # print(model1.cavSRY.round_trip_optical_length)
        print(model1.SRCL.DC.value)
        if model1.SRCL.DC.value > 0:
            label = f"SRC ∂L {(model1.SRCL.DC*1064/360) - 266:.3f} nm"
        else:
            label = f"SRC ∂L {(model1.SRCL.DC*1064/360) + 266:.3f} nm"
        print(label)
        # print((model1.cavSRX.gouy+model1.cavSRY.gouy)/2)
        data_output['gouy_shift'].append(model1.SRM.p1.i.qy.gouy() * 180/np.pi )
        data_output['SRCL_detuning'].append(model1.SRCL.DC.value)
        if detectors_on==True:
            data_output['HG00_SB'].append(out['a_l45_00_src'])
            data_output['HG00_amp'].append(out['HG00_amp'])
            data_output['HG10_amp'].append(out['HG10_amp'])
            data_output['HG20_amp'].append(out['HG20_amp'])
            data_output['HG01_amp'].append(out['HG01_amp'])
            data_output['HG02_amp'].append(out['HG02_amp'])
        sol =model1.run(fa.Series(fa.FrequencyResponse(F_Hz, model1.DARM.AC.i, model1.AS.DC.o, name="fresp")))
        plt.loglog(F_Hz, np.abs(sol['AS.DC.o', 'DARM.AC.i']), color=color, label=label)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [AS/DARM]")
plt.title("DARM sensing function")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/DARM_sensing_function.pdf")    
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/DARM_sensing_function.png")
plt.show()
#%%

if detectors_on==True:
    plt.plot(SRCL_DC_range,np.angle(data_output['HG00_SB'])*180/np.pi,'o',label="HG00_SB")
    plt.plot(SRCL_DC_range,np.angle(data_output['HG00_amp'])*180/np.pi,'o',label="HG00")
    plt.plot(SRCL_DC_range,np.angle(data_output['HG10_amp'])*180/np.pi,'x',label="HG10")
    plt.plot(SRCL_DC_range,np.angle(data_output['HG20_amp'])*180/np.pi,'d',label="HG20")
    plt.plot(SRCL_DC_range,np.angle(data_output['HG01_amp'])*180/np.pi,'s',label="HG01")
    plt.plot(SRCL_DC_range,np.angle(data_output['HG02_amp'])*180/np.pi,'v',label="HG02")
    plt.xlabel("SRM tilt [urad]")
    plt.ylabel("Phase [deg]")
    plt.title("Phase of Carrier HG00, HG10, HG20, HG01, HG02")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plt.plot(SRCL_DC_range,data_output['gouy_shift'])
plt.xlabel("SRM tilt [urad]")
plt.ylabel("Gouy shift [deg]")
plt.title("Gouy shift vs SRM tilt")
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
model2=factory.make()
model2.modes(maxtem=4) #try modes("off")
model2.fsig.f=1
minimize_MM(model2)
# model2.ITMX.ybeta=1e-9
detectors_on=False
if detectors_on==True:
    add_amplitude_detectors(model2)
cav_mismatches1=get_cav_mismatches(model2,print_tables=False)

model2.run(fac.Series(
(aligo.InitialLock()),
aligo.DARM_RF_to_DC()
))
SRCL_DC_range=np.linspace(-0.005,0.005,5)
colors = plt.cm.rainbow(np.linspace(0, 1, len(SRCL_DC_range)))
data_output={'HG00_SB':[],'HG00_amp':[],'HG10_amp':[],'HG20_amp':[],'HG01_amp':[],'HG02_amp':[],'SRCL_detuning':[],'gouy_shift':[]}
for i, color in zip(SRCL_DC_range, colors):
    with model2.temporary_parameters():
        # print(model2.SRCL.DC.value)
        model2.SR2.Rc *= (1+i)
        out=model2.run(fac.Noxaxis())
        cav_mismatches1=get_cav_mismatches(model2,print_tables=False)
        print(cav_mismatches1)

        if model2.SRCL.DC.value > 0:
            label = f"SRC ∂L {(model2.SRCL.DC*1064/360) - 266:.3f} nm"
        else:
            label = f"SRC ∂L {(model2.SRCL.DC*1064/360) + 266:.3f} nm"
        # print((model2.cavSRX.gouy+model2.cavSRY.gouy)/2)
        data_output['gouy_shift'].append(model2.SRM.p1.i.qy.gouy() * 180/np.pi )
        data_output['SRCL_detuning'].append(model2.SRCL.DC.value)
        if detectors_on==True:
            data_output['HG00_SB'].append(out['a_l45_00_src'])
            data_output['HG00_amp'].append(out['HG00_amp'])
            data_output['HG10_amp'].append(out['HG10_amp'])
            data_output['HG20_amp'].append(out['HG20_amp'])
            data_output['HG01_amp'].append(out['HG01_amp'])
            data_output['HG02_amp'].append(out['HG02_amp'])
        sol =model2.run(fa.Series(fa.FrequencyResponse(F_Hz, model2.DARM.AC.i, model2.AS.DC.o, name="fresp")))
        plt.loglog(F_Hz, np.abs(sol['AS.DC.o', 'DARM.AC.i']), color=color, label=label)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [AS/DARM]")
plt.title("DARM sensing function")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/DARM_sensing_function.pdf")    
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/DARM_sensing_function.png")
plt.show()
#%%




model3=factory.make()
model3.modes(maxtem=2) #try modes("off")
model3.fsig.f=1
# minimize_MM(model3)
detectors_on=True
if detectors_on==True:
    add_amplitude_detectors(model3)
cav_mismatches1=get_cav_mismatches(model3,print_tables=False)


out=model3.run(fac.Series(
(aligo.InitialLock()),
aligo.DARM_RF_to_DC(),
fac.Noxaxis(),
fac.Xaxis(parameter=model3.SRCL.DC,mode='lin',start=-45,stop=45,steps=50,name='SRM_phi',relative=True),
))

plt.plot(out['SRM_phi'].x1,np.abs(out['SRM_phi']['PRG45']/(out["SRM_phi"]['a_u45_00_src']*out["SRM_phi"]['a_l45_00_src']))**2)
plt.plot(out['SRM_phi'].x1,np.abs(out['SRM_phi']['cost_srcl'])**2,label='cost SRCL')
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [AS/DARM]")
plt.title("DARM sensing function")
plt.grid(True)
plt.legend()
plt.tight_layout()
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/DARM_sensing_function.pdf")    
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/DARM_sensing_function.png")
plt.show()
print(out['cost_srcl'], model3.SRCL.DC.value)
#%%

dc_vals = out['SRM_phi'].x1 
cost_vals = np.abs(out['SRM_phi']['cost_srcl'])**2

min_index = np.argmin(cost_vals)
min_dc = dc_vals[min_index]
min_cost = cost_vals[min_index]
srm_phi_at_min = out['SRM_phi']['cost_srcl'][min_index]  # assuming 'xvals' is your SRM_phi sweep array

print(f"Minimum |cost_srcl|^2: {min_cost}")
print(f"At SRCL.DC = {min_dc}")
print(f"SRM_phi value at minimum: {srm_phi_at_min}")

# %%
