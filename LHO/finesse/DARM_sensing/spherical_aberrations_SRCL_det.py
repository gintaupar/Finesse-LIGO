#%%

from more_itertools import roundrobin
from finesse.ligo.actions import InitialLockLIGO, DARM_RF_to_DC

import finesse.components as fc  

from munch import Munch
import finesse
finesse.init_plotting()
import finesse.detectors as det 

from finesse.ligo.factory import aligo
import finesse_ligo
import matplotlib.pyplot as plt
import scipy
import sys
# Add the analysis directory to sys.path
sys.path.append(str(finesse.ligo.git_path() / "analysis"/"O4"/"LHO"/"sensing_function"))

# Now you can import
from T_sensing_function import SQZ_metrics
import finesse.analysis.actions as fa
from functions import *

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
factory.options.thermal.add = False

F_Hz=np.geomspace(7, 1000, 100)


#%%

labels = []
magnitude_data = []
phase_data = []

pdh_data=[]
values={'Px':[],'Py':[],'SRCL_DC':[],'absorptions':[]}

absorptions=np.linspace(0.2e-6,0.5e-6,5) #absorption ppm
colors = plt.cm.rainbow(np.linspace(0, 1, len(absorptions)))
for i_idx, i in enumerate(absorptions):
    model1 = factory.make()
    # model1.modes(maxtem=4)
    model1.fsig.f=1
    minimize_MM(model1)
    # model1.L0.P=1
    model1.modes('even',maxtem=6)
    # model1.L0.P=i
    absorption_difference=i-absorptions[-(i_idx+1)]
    print("Absorption Difference $\alpha_{ITMX} - \alpha_{ITMY} : $", absorption_difference)
    print("Absorption ITMX", i)
    print("Absorption ITMY", absorptions[-(i_idx+1)])

    outputs =make_thermal_lens(model1,arms=["X","Y"],plot=False,absorption_x=i,absorption_y=absorptions[-(i_idx+1)])
    try:
        model1.run(fa.Series(aligo.InitialLock(),aligo.DARM_RF_to_DC(),fa.Noxaxis()))
    except:
        print("Initial Lock Failed")
        continue

    values['Px'].append(outputs["Px"])
    values['Py'].append(outputs["Py"])
    values['SRCL_DC'].append(model1.SRCL.DC)
    values['absorptions'].append(i)
    print(model1.SRCL.DC)
    sol = model1.run(
    fa.Series(fa.FrequencyResponse(F_Hz, model1.DARM.AC.i, model1.AS.DC.o, name="fresp")))

    label = f"{(absorption_difference*1e6):.2f} ppm"
    labels.append(label)
    target = 100
    index_near_100 = np.argmin(np.abs(F_Hz - target))
    magnitude_data.append(np.abs(sol['AS.DC.o', 'DARM.AC.i']))
    phase_data.append(np.angle(sol['AS.DC.o', 'DARM.AC.i']) * 180 / np.pi)



#%%
values_at_100Hz=[]
Pin_avg = (np.array(values['Px']) + np.array(values['Py'])) / 2

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

for mag, phase, label, color in zip(magnitude_data, phase_data, labels, colors):
    ax1.loglog(F_Hz, mag, label=label, color=color)
    ax2.semilogx(F_Hz, phase, label=label, color=color)
    values_at_100Hz.append(mag[index_near_100])

ax1.set_title(f"Thermal Distortion Effects on DARM Sensing Function with Absorption ITMX - ITMY")

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
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/sensing_function_with_thermal_distortion.pdf")
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/sensing_function_with_thermal_distortion.png")
plt.show()
# plt.plot(start_detuning+SRCL_detuning,powers['DHARD_P_DC'],label='Power in XARM')
# plt.xlabel("Detuning Angle [deg]")
# plt.ylabel("Power inside XARM [W]")
# plt.legend()
# plt.grid(True)
# plt.show()

#turned off for now because I'm not chaning the power in the model
# plot the sensing function value at around 100 Hz vs input power... it should be a square root shape

# def sqrt_model(x, a, b):
#     return a * np.sqrt(x) + b

# # Fit the model
# params, _ = scipy.optimize.curve_fit(sqrt_model, Input_powers, values_at_100Hz,p0=[1.7e9,values_at_100Hz[0]])
# a, b = params

# # Generate smooth x values for plotting the fit
# x_fit = np.linspace(min(Input_powers), max(Input_powers), 500)
# y_fit = sqrt_model(x_fit, a, b)

# # Plot the original data and the fit
# plt.plot(Input_powers, values_at_100Hz, 'bo', label='Data')
# plt.plot(x_fit, y_fit, 'r--', label=f'Fit: y = {a:.3f}√x + {b:.3f}')
# plt.xlabel('Input Power')
# plt.ylabel('Value at 100 Hz')
# plt.title("Detector's sensitivity as Input Power Increase ")
# plt.grid(True)
# plt.legend()
# plt.show()

#%%


SRCL_DC_vals = wrap_angle(np.abs(values['SRCL_DC']))

Pin_pct_change = 100 * (Pin_avg - Pin_avg[0]) / Pin_avg[0]
SRCL_pct_change = 100 * (np.abs(values['SRCL_DC']) - np.abs(values['SRCL_DC'][0])) / np.abs(values['SRCL_DC'][0])

fig, ax1 = plt.subplots(figsize=(6,4))

# First y-axis: SRCL DC vs Pin (kW)
color1 = 'tab:blue'
plt.xlabel("Circulating Power in Arms [kW]")
plt.ylabel("SRCL.DC", color=color1)
plt.plot(values['absorptions'], wrap_angle(np.array(values['SRCL_DC'])), 'o-', label='SRCL DC Offset', color=colors[0])
plt.tick_params(axis='y', labelcolor=color1)

# # Second y-axis: % change of SRCL DC vs % change of Power
# ax2 = ax1.twinx()  # create a second y-axis sharing the same x-axis
# color2 = 'tab:red'
# ax2.set_ylabel("ΔSRCL.DC [%]", color=color2)
# ax2.plot(Pin_avg / 1e3, SRCL_pct_change, 's', label='Relative Change', color=color2)
# ax2.tick_params(axis='y', labelcolor=color2)

# Title and layout
plt.title("Thermal Distortion Effects on SRCL DC Detuning with Circulating Power")
fig.tight_layout()
plt.grid(True)
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/SRCL_detuning_with_thermal_distortion.pdf")
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/SRCL_detuning_with_thermal_distortion.png")

plt.show()

# %%



labels = []
magnitude_data = []
phase_data = []

pdh_data=[]
values={'Px':[],'Py':[],'SRCL_DC':[],'absorptions':[]}

absorptions=np.linspace(0e-6,0.7e-6,5) #absorption ppm
colors = plt.cm.rainbow(np.linspace(0, 1, len(absorptions)))
for i_idx, i in enumerate(absorptions):
    model1 = factory.make()
    # model1.modes(maxtem=4)
    model1.fsig.f=1
    minimize_MM(model1)
    # model1.L0.P=1
    model1.modes('even',maxtem=6)
    # model1.L0.P=50

    print("Absorption is", i)

    outputs =make_thermal_lens(model1,arms=["X","Y"],plot=False,absorption_x=i,absorption_y=i)
    try:
        out=model1.run(fa.Series(aligo.InitialLock(),aligo.DARM_RF_to_DC(),fa.Noxaxis()))
    except:
        print("Initial Lock Failed")
        continue

    values['Px'].append(outputs["Px"])
    values['Py'].append(outputs["Py"])
    values['SRCL_DC'].append(model1.SRCL.DC)
    values['absorptions'].append(i)
    print(model1.SRCL.DC)
    sol = model1.run(
    fa.Series(fa.FrequencyResponse(F_Hz, model1.DARM.AC.i, model1.AS.DC.o, name="fresp")))

    label = f"{(i*1e6):.2f} ppm"
    labels.append(label)
    target = 100
    index_near_100 = np.argmin(np.abs(F_Hz - target))
    magnitude_data.append(np.abs(sol['AS.DC.o', 'DARM.AC.i']))
    phase_data.append(np.angle(sol['AS.DC.o', 'DARM.AC.i']) * 180 / np.pi)



#%%
values_at_100Hz=[]
Pin_avg = (np.array(values['Px']) + np.array(values['Py'])) / 2

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

for mag, phase, label, color in zip(magnitude_data, phase_data, labels, colors):
    ax1.loglog(F_Hz, mag, label=label, color=color)
    ax2.semilogx(F_Hz, phase, label=label, color=color)
    values_at_100Hz.append(mag[index_near_100])

ax1.set_title(f"Thermal Distortion Effects on DARM Sensing Function with Absorption ITMX = ITMY")

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
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/sensing_function_with_thermal_distortion.pdf")
# plt.savefig(f"{parentrepo}/LHO/finesse/DARM_sensing/plots/sensing_function_with_thermal_distortion.png")
plt.show()
# plt.plot(start_detuning+SRCL_detuning,powers['DHARD_P_DC'],label='Power in XARM')
# plt.xlabel("Detuning Angle [deg]")
# plt.ylabel("Power inside XARM [W]")
# plt.legend()
# plt.grid(True)
# plt.show()
# %%
