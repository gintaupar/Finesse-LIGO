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
import pandas as pd
from finesse.ligo.ASC_controllers import get_controller
from finesse.ligo.suspension import QUADSuspension
from scipy.integrate import *
import scipy
import finesse.components as fc
from IPython.display import Image
import finesse.detectors as det
from finesse.analysis.actions import FrequencyResponse
import cmath
  
from sympy import integrate
from finesse.analysis.actions.axes import Noxaxis
finesse.init_plotting()
# from control import tf, root_locus
from finesse.components.mechanical import Pendulum
import finesse_ligo

# Generate tables for documentation
from functions import *


# finesse.plot_
#let's plot OLG TF
parentrepo = f'{finesse.ligo.git_path()}/' #str(finesse.ligo.git_path())

factory = aligo.ALIGOFactory(f"{parentrepo}/LHO/yaml/lho_O4.yaml")
# factory.reset()
# factory.options.QUAD_suspension_model = QUADSuspension
factory.options.QUAD_suspension_model = finesse_ligo.suspension.QUADSuspension
# factory.options.QUAD_suspension_model = fc.FreeMass
# factory.options.QUAD_suspension_kwargs = dict(mass=40)
factory.options.ASC.add = True
factory.options.ASC.add_DOFs = True
factory.options.ASC.add_locks=True
factory.options.ASC.add_output_detectors=True
factory.options.ASC.add_readouts = True
factory.options.ASC.add_AC_loops = True
factory.options.ASC.close_AC_loops = True  


F_Hz=np.geomspace(7, 1000, 50)

# %%

detector_names = [
    'HG20_SRC_carrier', 'HG02_SRC_carrier',
    'HG20_SRC_45', 'HG02_SRC_45',
    'HG20_SRC_9', 'HG02_SRC_9'
]

# Create a dictionary of zeroed arrays to hold the detector values

detector_data={name:[] for name in detector_names}


model1 = factory.make()
model1.modes(maxtem=4)
# model1.remove(model1.gIM2_p1_i)

cav_mismatches1=get_cav_mismatches(model1,print_tables=False)

model1.fsig.f = 1 

# model1.L0.P=1
###### RF #####
model1.run(fac.Series(
(aligo.InitialLock()),
Noxaxis(),
))
# model2.DARM.DC=0
sol1 = model1.run(
fa.Series(fa.FrequencyResponse(F_Hz, model1.DARM.AC.i, model1.AS45.Q.o, name="sol1")))
print(f"RF Avg MM {100*cav_mismatches1['total_avg']:.2f}% ","DARM.DC", model1.DARM.DC,"SRCL.DC", model1.SRCL.DC.value)


###### DC #####
model1.run(aligo.DARM_RF_to_DC())

sol3 = model1.run(
fa.Series(fa.FrequencyResponse(F_Hz, model1.DARM.AC.i, model1.AS.DC.o, name="sol1")))
print(f"DC Avg MM {100*cav_mismatches1['total_avg']:.2f}% ","DARM.DC",model1.DARM.DC,"SRCL.DC", model1.SRCL.DC.value)

fig, ax1 = plt.subplots()

# First y-axis: sol1 and sol2
plt.loglog(F_Hz, np.abs(sol1['AS45.Q.o', 'DARM.AC.i']), label=f"RF Avg MM {100*cav_mismatches1['total_avg']:.2f}%")
plt.loglog(F_Hz, np.abs(sol3['AS.DC.o', 'DARM.AC.i']), label=f"DC Avg MM {100*cav_mismatches1['total_avg']:.2f}%")

# model1.ETMY.Rc=2210.7

model1 = factory.make()
model1.modes(maxtem=4)
model1.fsig.f=1
minimize_MM(model1)
# model1.remove(model1.gIM2_p1_i)


# print(model1.mismatches_table())
cav_mismatches2=get_cav_mismatches(model1,print_tables=False)

###### RF #####

model1.run(fac.Series(
(aligo.InitialLock()),
Noxaxis(),
))
print(f"RF Avg MM {100*cav_mismatches2['total_avg']:.2f}% ","DARM.DC",model1.DARM.DC,"SRCL.DC", model1.SRCL.DC.value)

sol2 = model1.run(
fa.Series(fa.FrequencyResponse(F_Hz, model1.DARM.AC.i, model1.AS45.Q.o, name="sol2")))

###### DC #####
model1.run(aligo.DARM_RF_to_DC())
print(f"DC Avg MM {100*cav_mismatches2['total_avg']:.2f}% ","DARM.DC",model1.DARM.DC,"SRCL.DC", model1.SRCL.DC.value)
sol4 = model1.run(
fa.Series(fa.FrequencyResponse(F_Hz, model1.DARM.AC.i, model1.AS.DC.o, name="sol2")))

plt.loglog(F_Hz, np.abs(sol2['AS45.Q.o', 'DARM.AC.i']), label=f"RF Avg MM {100*cav_mismatches2['total_avg']:.2f}%")
plt.loglog(F_Hz, np.abs(sol4['AS.DC.o', 'DARM.AC.i']), label=f"DC Avg MM {100*cav_mismatches2['total_avg']:.2f}%")

plt.xlabel("Frequency")
plt.ylabel("Magnitude")

plt.legend()
plt.tight_layout()
plt.title(f"DARM input->DARM readout sensing function")
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/Sensing_function_improved_mm.pdf")
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/Sensing_function_improved_mm.png")
plt.show()


# table=mismatch_y
#   
# df = pd.DataFrame(table.table)
# from matplotlib.backends.backend_pdf import PdfPages

# # Create a figure and axis
# fig, ax = plt.subplots(figsize=(8, 2))
# ax.axis('off')  # Hide the axes

# # Create table
# table = ax.table(cellText=df.values,
#                  colLabels=df.columns,
#                  cellLoc='center',
#                  loc='center')

# table.scale(1, 1.5)  # Scale table size

# # Save to PDF
# # with PdfPages(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/{table}.pdf") as pdf:
#     # pdf.savefig(fig, bbox_inches='tight')

#%%

model2 = factory.make()
model2.modes(maxtem=4)
model2.fsig.f=1
minimize_MM(model2)
# model2.remove(model2.gIM2_p1_i)
# print(model2.mismatches_table())
cav_mismatches2=get_cav_mismatches(model2,print_tables=False)
# print()
###### DC #####

model2.run(fac.Series(
(aligo.InitialLock()),
aligo.DARM_RF_to_DC(),
Noxaxis(),
))
print(f"DC Avg MM {100*cav_mismatches2['total_avg']:.2f}% ","DARM.DC",model2.DARM.DC,"SRCL.DC", model2.SRCL.DC.value)

sol1 = model2.run(
fa.Series(fa.FrequencyResponse(F_Hz, model2.DARM.AC.i, model2.AS.DC.o, name="sol2")))

###### DC #####
model2.DARM.DC=model2.DARM.DC/10
print(f"DC Avg MM {100*cav_mismatches2['total_avg']:.2f}% ","DARM.DC",model2.DARM.DC,"SRCL.DC", model2.SRCL.DC.value)
sol2 = model2.run(
fa.Series(fa.FrequencyResponse(F_Hz, model2.DARM.AC.i, model2.AS.DC.o, name="sol2")))

#%%
plt.loglog(F_Hz,np.abs(sol1['AS.DC.o', 'DARM.AC.i']), label=f"DARM DC {10*model2.DARM.DC.value/1e-6:.3f}µDeg")
plt.loglog(F_Hz,np.abs(sol2['AS.DC.o', 'DARM.AC.i']), '-',label=f"DARM DC {model2.DARM.DC.value/1e-6:.3f}µDeg")

plt.xlabel("Frequency")
plt.ylabel("Magnitude")

plt.legend()
plt.tight_layout()
plt.title(f"DARM input->DARM readout vs DARM DC")
# # plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/1W_DARM_readout_vs_detuning.pdf")
# # plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/1W_DARM_readout_vs_detuning.png")
plt.show()
#%%


#### The effect of ITM change vs ETM
steps=5
# ITMXlens_range=np.linspace(-584332.3329218434,-144332.3329218434,steps)
# ITMYlens_range=np.linspace(478580.1553062986,178580.1553062986,steps)

ITMXlens_range=np.linspace(0.985,1.015,steps)
ITMYlens_range=np.linspace(1.015,0.985,steps)


# SRCL_DC_2D = np.zeros((len(ITMXlens_range), len(ITMYlens_range)))
labels = {'graph_labels':[]}
magnitude_data = []
phase_data = []

pdh_data=[]

data={'XYARM':[],'SRXXARM':[],'SRYYARM':[],'PRXXARM':[],'PRYYARM':[],'SRCL_DC':[],'ITMX_RoC_change':[],'ITMY_RoC_change':[],'XYARM_MM':[]
,'ITMX_w':[],'ITMY_w':[],'ETMX_w':[],'ETMY_w':[]}
for idx,i in enumerate(ITMXlens_range):
    print(idx)
        # factory.reset()
    model3 = factory.make()
    model3.fsig.f=1
    model3.modes(maxtem=4)
    minimize_MM(model3)

    model3.ITMX.Rc *= ITMXlens_range[idx]
    model3.ITMY.Rc *= ITMYlens_range[idx]

    # model3.ETMY.Rc *= ITMYlens_range[idx]
    # print(model3.ETMX.Rc)
    # print(model3.ITMXlens.f,model3.ITMYlens.f)
    # # model3.L0.P=1

    # model3.remove(model3.gIM2_p1_i)

    cav_mismatches2=get_cav_mismatches(model3,print_tables=False)

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

    print("XYARMs_MM",XYARMs_MM,"SRXXARMs_MM",SRXXARMs_MM,"SRYYARMs_MM",SRYYARMs_MM)
    print("PRXXARMs_MM",PRXXARMs_MM,"PRYYARMs_MM",PRYYARMs_MM)

    # print(model3.ITMXlens.f ,model3.ITMYlens.f)
    model3_out = model3.run(fac.Series(
        aligo.InitialLock(),
        aligo.DARM_RF_to_DC(),
        Noxaxis()
    ))
    print("SRCL.DC", model3.SRCL.DC)
    data['SRCL_DC'].append(model3.SRCL.DC.value)

    # model3.run(aligo.DARM_RF_to_DC())
    sol = model3.run(
    fa.Series(fa.FrequencyResponse(F_Hz, model3.DARM.AC.i, model3.AS.DC.o, name="fresp")))
    label = f"ITMX RoC: {((2/model3.ITMX.Rc)/1e-6)[0]:.2f} µD, ITMY RoC: {((2/model3.ITMY.Rc)/1e-6)[0]:.2f} µD" #f"{(XYARMs_MM/1e-2):.2f} %"
    labels['graph_labels'].append(label)
    data['ITMX_RoC_change'].append(model3.ITMX.Rc)
    data['ITMY_RoC_change'].append(model3.ITMY.Rc)
    data['XYARM_MM'].append(XYARMs_MM)
    data['ITMX_w'].append(model3.ITMX.p1.i.qx.w)
    data['ITMY_w'].append(model3.ITMY.p1.i.qx.w)
    data['ETMX_w'].append(model3.ETMX.p1.i.qx.w)
    data['ETMY_w'].append(model3.ETMY.p1.i.qx.w)
    target = 100
    index_near_100 = np.argmin(np.abs(F_Hz - target))
    magnitude_data.append(np.abs(sol['AS.DC.o', 'DARM.AC.i']))
    phase_data.append(np.angle(sol['AS.DC.o', 'DARM.AC.i']) * 180 / np.pi)

#%%
  

# Create tables for each analysis
print("\nGenerating tables for differential ITM RoC case...")
create_data_table(labels, data,name="diff_ITM_RoC", save_md=True, save_png=True)

print("\nTables have been generated:")
print("1. data_table.md - Markdown format table")
print("2. data_table.png - Visual table image")


# Assuming your data dictionary is already populated
df = pd.DataFrame(data)
 
     

df['SRCL_DC_wrapped'] = df['SRCL_DC'].apply(wrap_angle)
avg_MM=(data['XYARM']/(np.array(data['SRXXARM'])+np.array(data['SRYYARM'])))

# Plot
plt.figure(figsize=(6, 4))
sc = plt.scatter(100*np.array(data['XYARM']), df['SRCL_DC_wrapped'], c=avg_MM, cmap='plasma', s=100, edgecolor='k')
cbar = plt.colorbar(sc)
cbar.set_label('XARM-YARM MM /SRC-ARMs_MM', rotation=270, labelpad=15)

plt.xlabel('XARM-YARM Mismatch %')
plt.ylabel('SRCL DC')
plt.title('Differential ITM RoC')
plt.grid(True)
plt.tight_layout()
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SRCL_detuning_diff_ITM_RoC.pdf")
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SRCL_detuning_diff_ITM_RoC.png")
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

for mag, phase, label in zip(magnitude_data, phase_data, labels['graph_labels']):
    ax1.loglog(F_Hz, mag, label=label)
    ax2.semilogx(F_Hz, phase, label=label)
ax1.set_title(f"DARM input->DARM readout at Differential ITM RoC")

ax1.set_ylabel("Magnitude [AS/DARM]")
# ax1.legend()
ax1.grid(True, which='both')

ax2.set_ylabel("Phase [deg]")
ax2.set_xlabel("Frequency [Hz]")
ax2.legend()
ax2.grid(True, which='both')

# plt.legend()
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Ampltiude [AS/DARM]")
# plt.grid(True)
plt.tight_layout()
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SF_vs_diff_ITM_RoC.pdf")
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SF_vs_diff_ITM_RoC.png")
plt.show()

#%%


###### checking the change in lens to mismatch XYARMs
steps=5

ITMXlens_range=np.linspace(0.7,0.2,steps)
ITMYlens_range=np.linspace(1,0.2,steps)


# SRCL_DC_2D = np.zeros((len(ITMXlens_range), len(ITMYlens_range)))
labels = {'graph_labels':[]}
magnitude_data = []
phase_data = []

pdh_data=[]

data={'XYARM':[],'SRXXARM':[],'SRYYARM':[],'PRXXARM':[],'PRYYARM':[],'SRCL_DC':[],'ITMX_lens_change':[],'ITMY_lens_change':[],'XYARM_MM':[]
,'ITMX_w':[],'ITMY_w':[],'ETMX_w':[],'ETMY_w':[]}
for idx,i in enumerate(ITMXlens_range):
    print(idx)
        # factory.reset()
    model4 = factory.make()
    model4.fsig.f=1
    # model4.SRM.R=0.8 

    model4.modes(maxtem=4)
    minimize_MM(model4)
    k = 0.0003  # compresses the 2% range down to 0.02%
    scale = 1 + (ITMYlens_range[idx] - 1) * k
    # model4.SR2.Rc *= scale
    # model4.SR2.Rc=-6.4135*scale
    # model4.SRM.Rc=np.array([-5.623,-5.763])*scale

    model4.ITMXlens.f *= ITMXlens_range[idx]
    model4.ITMYlens.f *= ITMYlens_range[idx]
    # print(model4.ETMX.Rc)
    # print(model4.ITMXlens.f,model4.ITMYlens.f)
    # # model4.L0.P=1

    # model4.remove(model4.gIM2_p1_i)

    cav_mismatches2=get_cav_mismatches(model4,print_tables=False)

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

    print("XYARMs_MM",XYARMs_MM,"SRXXARMs_MM",SRXXARMs_MM,"SRYYARMs_MM",SRYYARMs_MM)
    print("PRXXARMs_MM",PRXXARMs_MM,"PRYYARMs_MM",PRYYARMs_MM)

    # print(model4.ITMXlens.f ,model4.ITMYlens.f)
    model4_out = model4.run(fac.Series(
        aligo.InitialLock(),
        aligo.DARM_RF_to_DC(),
        Noxaxis()
    ))
    print("SRCL.DC", model4.SRCL.DC)
    data['SRCL_DC'].append(model4.SRCL.DC.value)

    # model4.run(aligo.DARM_RF_to_DC())
    sol = model4.run(
    fa.Series(fa.FrequencyResponse(F_Hz, model4.DARM.AC.i, model4.AS.DC.o, name="fresp")))
    label = f"ITMXlens: {(1/model4.ITMXlens.f)/1e-6:.2f} µD, ITMYlens: {(1/model4.ITMYlens.f)/1e-6:.2f} µD" #f"{(XYARMs_MM/1e-2):.2f} %"
    labels['graph_labels'].append(label)
    data['ITMX_lens_change'].append(model4.ITMXlens.f)
    data['ITMY_lens_change'].append(model4.ITMYlens.f)
    data['XYARM_MM'].append(XYARMs_MM)
    data['ITMX_w'].append(model4.ITMX.p1.i.qx.w)
    data['ITMY_w'].append(model4.ITMY.p1.i.qx.w)
    data['ETMX_w'].append(model4.ETMX.p1.i.qx.w)
    data['ETMY_w'].append(model4.ETMY.p1.i.qx.w)
    target = 100
    index_near_100 = np.argmin(np.abs(F_Hz - target))
    magnitude_data.append(np.abs(sol['AS.DC.o', 'DARM.AC.i']))
    phase_data.append(np.angle(sol['AS.DC.o', 'DARM.AC.i']) * 180 / np.pi)


#%%

# Create tables for each analysis
print("\nGenerating tables for differential ITM lens case...")
create_data_table(labels, data,name="diff_ITM_lens", save_md=True, save_png=True)



# Assuming your data dictionary is already populated
df = pd.DataFrame(data)

df['SRCL_DC_wrapped'] = df['SRCL_DC'].apply(wrap_angle)
avg_MM=(data['XYARM']/(np.array(data['SRXXARM'])+np.array(data['SRYYARM'])))

# Plot
plt.figure(figsize=(6, 4))
sc = plt.scatter(100*np.array(data['XYARM']), df['SRCL_DC_wrapped'], c=avg_MM, cmap='plasma', s=100, edgecolor='k')
cbar = plt.colorbar(sc)
cbar.set_label('XARM-YARM MM /SRC-ARMs_MM', rotation=270, labelpad=15)

plt.xlabel('XARM-YARM Mismatch %')
plt.ylabel('SRCL DC')
plt.title('Differential ITM lens')
plt.grid(True)
plt.tight_layout()
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SRCL_detuning_diff_ITM_lens.pdf")
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SRCL_detuning_diff_ITM_lens.png")
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

for mag, phase, label in zip(magnitude_data, phase_data, labels['graph_labels']):
    ax1.loglog(F_Hz, mag, label=label)
    ax2.semilogx(F_Hz, phase, label=label)
ax1.set_title(f"DARM input->DARM readout at Differential ITM lens")

ax1.set_ylabel("Magnitude [AS/DARM]")
# ax1.legend()
ax1.grid(True, which='both')

ax2.set_ylabel("Phase [deg]")
ax2.set_xlabel("Frequency [Hz]")
ax2.legend()
ax2.grid(True, which='both')

# plt.legend()
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Ampltiude [AS/DARM]")
# plt.grid(True)
plt.tight_layout()
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SF_vs_diff_ITM_lens.pdf")
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SF_vs_diff_ITM_lens.png")
plt.show()



#%%

### The effect of ETM radius change rather than ITM or lens
steps=5
# ITMXlens_range=np.linspace(-584332.3329218434,-144332.3329218434,steps)
# ITMYlens_range=np.linspace(478580.1553062986,178580.1553062986,steps)

ITMXlens_range=np.linspace(0.985,1.015,steps)
ITMYlens_range=np.linspace(1.015,0.985,steps)


# SRCL_DC_2D = np.zeros((len(ITMXlens_range), len(ITMYlens_range)))
labels = {'graph_labels':[]}
magnitude_data = []
phase_data = []

pdh_data=[]

data={'XYARM':[],'SRXXARM':[],'SRYYARM':[],'PRXXARM':[],'PRYYARM':[],'SRCL_DC':[],'ITMX_lens_change':[],'ITMY_lens_change':[],'XYARM_MM':[]
,'ITMX_w':[],'ITMY_w':[],'ETMX_w':[],'ETMY_w':[]}
for idx,i in enumerate(ITMXlens_range):
    print(idx)
        # factory.reset()
    model5 = factory.make()
    model5.fsig.f=1
    model5.modes(maxtem=4)
    minimize_MM(model5)
    k = 0.001  # compresses the 2% range down to 0.02%
    scale = 1 + (ITMXlens_range[idx] - 1) * k
    # model3.SR2.Rc *= scale
    model5.SR2.Rc=-6.4135*scale
    model5.SRM.Rc=np.array([-5.623,-5.763])*scale

    model5.ETMX.Rc *= ITMXlens_range[idx]
    model5.ETMY.Rc *= ITMYlens_range[idx]
    print(model5.ETMX.Rc)
    # print(model3.ITMXlens.f,model3.ITMYlens.f)
    # # model3.L0.P=1

    # model3.remove(model3.gIM2_p1_i)

    cav_mismatches2=get_cav_mismatches(model5,print_tables=False)

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

    print("XYARMs_MM",XYARMs_MM,"SRXXARMs_MM",SRXXARMs_MM,"SRYYARMs_MM",SRYYARMs_MM)
    print("PRXXARMs_MM",PRXXARMs_MM,"PRYYARMs_MM",PRYYARMs_MM)

    # print(model3.ITMXlens.f ,model3.ITMYlens.f)
    model5_out = model5.run(fac.Series(
        aligo.InitialLock(),
        aligo.DARM_RF_to_DC(),
        Noxaxis()
    ))
    print("SRCL.DC", model5.SRCL.DC)
    data['SRCL_DC'].append(model5.SRCL.DC.value)

    # model3.run(aligo.DARM_RF_to_DC())
    sol = model5.run(
    fa.Series(fa.FrequencyResponse(F_Hz, model5.DARM.AC.i, model5.AS.DC.o, name="fresp")))
    label = f"ETMX RoC: {((2/model5.ETMX.Rc)/1e-6)[0]:.2f} µD, ETMY RoC: {((2/model5.ETMY.Rc)/1e-6)[0]:.2f} µD" #f"{(XYARMs_MM/1e-2):.2f} %"
    labels['graph_labels'].append(label)
    data['ITMX_lens_change'].append(model5.ETMX.Rc)
    data['ITMY_lens_change'].append(model5.ETMY.Rc)
    data['XYARM_MM'].append(XYARMs_MM)
    data['ITMX_w'].append(model5.ETMX.p1.i.qx.w)
    data['ITMY_w'].append(model5.ETMY.p1.i.qx.w)
    data['ETMX_w'].append(model5.ETMX.p1.i.qx.w)
    data['ETMY_w'].append(model5.ETMY.p1.i.qx.w)
    target = 100
    index_near_100 = np.argmin(np.abs(F_Hz - target))
    magnitude_data.append(np.abs(sol['AS.DC.o', 'DARM.AC.i']))
    phase_data.append(np.angle(sol['AS.DC.o', 'DARM.AC.i']) * 180 / np.pi)

#%%
  
create_data_table(labels, data,name="diff_ETM_RoC", save_md=True, save_png=True)
# Assuming your data dictionary is already populated
df = pd.DataFrame(data)
 
     

df['SRCL_DC_wrapped'] = df['SRCL_DC'].apply(wrap_angle)
avg_MM=(data['XYARM']/(np.array(data['SRXXARM'])+np.array(data['SRYYARM'])))

# Plot
plt.figure(figsize=(6, 4))
sc = plt.scatter(100*np.array(data['XYARM']), df['SRCL_DC_wrapped'], c=avg_MM, cmap='plasma', s=100, edgecolor='k')
cbar = plt.colorbar(sc)
cbar.set_label('XARM-YARM MM /SRC-ARMs_MM', rotation=270, labelpad=15)

plt.xlabel('XARM-YARM Mismatch %')
plt.ylabel('SRCL DC')
plt.title('Differential ETM RoC')
plt.grid(True)
plt.tight_layout()
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SRCL_detuning_diff_ETM_RoC.pdf")
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SRCL_detuning_diff_ETM_RoC.png")
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

for mag, phase, label in zip(magnitude_data, phase_data, labels['graph_labels']):
    ax1.loglog(F_Hz, mag, label=label)
    ax2.semilogx(F_Hz, phase, label=label)
ax1.set_title(f"DARM input->DARM readout Differential ETM RoC")

ax1.set_ylabel("Magnitude [AS/DARM]")
# ax1.legend()
ax1.grid(True, which='both')

ax2.set_ylabel("Phase [deg]")
ax2.set_xlabel("Frequency [Hz]")
ax2.legend()
ax2.grid(True, which='both')

# plt.legend()
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Ampltiude [AS/DARM]")
# plt.grid(True)
plt.tight_layout()
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SF_vs_diff_ETM_RoC.pdf")
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SF_vs_diff_ETM_RoC.png")
plt.show()



#%%

#### now let's look at the case of same XYARM mismatch 
#but SRC/ARMs mismatch changes 
steps=5
# ITMXlens_range=np.linspace(-584332.3329218434,-144332.3329218434,steps)
# ITMYlens_range=np.linspace(478580.1553062986,178580.1553062986,steps)

ITMXlens_range=np.linspace(0.95,1.05,steps)
ITMYlens_range=np.linspace(0.95,1.05,steps)


# SRCL_DC_2D = np.zeros((len(ITMXlens_range), len(ITMYlens_range)))
labels = {'graph_labels':[]}
magnitude_data = []
phase_data = []

pdh_data=[]

data={'XYARM':[],'SRXXARM':[],'SRYYARM':[],'PRXXARM':[],'PRYYARM':[],'SRCL_DC':[],'ITMX_lens_change':[],'ITMY_lens_change':[],'XYARM_MM':[]
,'ITMX_w':[],'ITMY_w':[],'ETMX_w':[],'ETMY_w':[]}
for idx,i in enumerate(ITMXlens_range):
    print(idx)
        # factory.reset()
    model6 = factory.make()
    model6.fsig.f=1
    model6.modes(maxtem=6)

    minimize_MM(model6)

    model6.ITMX.Rc *= ITMXlens_range[idx]
    model6.ITMY.Rc *= ITMYlens_range[idx]
    # print(model4.ITMXlens.f,model4.ITMYlens.f)
    # # model4.L0.P=1

    # model4.remove(model4.gIM2_p1_i)

    cav_mismatches2=get_cav_mismatches(model6,print_tables=False)

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

    print("XYARMs_MM",XYARMs_MM,"SRXXARMs_MM",SRXXARMs_MM,"SRYYARMs_MM",SRYYARMs_MM)
    print("PRXXARMs_MM",PRXXARMs_MM,"PRYYARMs_MM",PRYYARMs_MM)

    # print(model4.ITMXlens.f ,model4.ITMYlens.f)
    model6_out = model6.run(fac.Series(
        aligo.InitialLock(),
        aligo.DARM_RF_to_DC(),
        Noxaxis()
    ))
    print("SRCL.DC", model6.SRCL.DC)
    data['SRCL_DC'].append(model6.SRCL.DC.value)

    # model4.run(aligo.DARM_RF_to_DC())
    sol = model6.run(
    fa.Series(fa.FrequencyResponse(F_Hz, model6.DARM.AC.i, model6.AS.DC.o, name="fresp")))
    label = f"ITMX RoC: {((2/model6.ITMX.Rc)/1e-6)[0]:.2f} µD, ITMY RoC: {((2/model6.ITMY.Rc)/1e-6)[0]:.2f} µD" #f"{(XYARMs_MM/1e-2):.2f} %"
    labels['graph_labels'].append(label)
    data['ITMX_lens_change'].append(model6.ITMX.Rc)
    data['ITMY_lens_change'].append(model6.ITMY.Rc)
    data['XYARM_MM'].append(XYARMs_MM)
    data['ITMX_w'].append(model6.ITMX.p1.i.qx.w)
    data['ITMY_w'].append(model6.ITMY.p1.i.qx.w)
    data['ETMX_w'].append(model6.ETMX.p1.i.qx.w)
    data['ETMY_w'].append(model6.ETMY.p1.i.qx.w)
    target = 100
    index_near_100 = np.argmin(np.abs(F_Hz - target))
    magnitude_data.append(np.abs(sol['AS.DC.o', 'DARM.AC.i']))
    phase_data.append(np.angle(sol['AS.DC.o', 'DARM.AC.i']) * 180 / np.pi)

#%%
create_data_table(labels, data,name="common_ITM_RoC", save_md=True, save_png=True)

# Assuming your data dictionary is already populated
df = pd.DataFrame(data)
 

SRC_ARM_MM=(np.array(data['SRXXARM'])+np.array(data['SRYYARM']))/2
df['SRCL_DC_wrapped'] = df['SRCL_DC'].apply(wrap_angle)
avg_MM=(SRC_ARM_MM/data['XYARM'])

# Plot
plt.figure(figsize=(6, 4))
sc = plt.scatter(100*SRC_ARM_MM, df['SRCL_DC_wrapped'], c=avg_MM, cmap='plasma', s=100, edgecolor='k')
cbar = plt.colorbar(sc)
cbar.set_label('SRC-ARMs MM/ XARM-YARM MM', rotation=270, labelpad=15)

plt.xlabel('SRC-ARMs Mismatch %')
plt.ylabel('SRCL DC')
plt.title('Common ITM RoC')
plt.grid(True)
plt.tight_layout()
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SRCL_detuning_common_ITM_RoC.pdf")
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SRCL_detuning_common_ITM_RoC.png")
plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

for mag, phase, label in zip(magnitude_data, phase_data, labels['graph_labels']):
    ax1.loglog(F_Hz, mag, label=label)
    ax2.semilogx(F_Hz, phase, label=label)
ax1.set_title(f"DARM input->DARM readout at Common ITM RoC")

ax1.set_ylabel("Magnitude [AS/DARM]")
# ax1.legend()
ax1.grid(True, which='both')

ax2.set_ylabel("Phase [deg]")
ax2.set_xlabel("Frequency [Hz]")
ax2.legend()
ax2.grid(True, which='both')

# plt.legend()
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Ampltiude [AS/DARM]")
# plt.grid(True)
plt.tight_layout()
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SF_vs_common_ITM_RoC.pdf")
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/perfect_mm/SF_vs_common_ITM_RoC.png")
plt.show()



# %%
