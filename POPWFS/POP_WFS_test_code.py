'''Test out new POP WFS telescope and 36 and 45 MHz sensitivity to carrier loss
Following work done in https://dcc.ligo.org/LIGO-G1500519
First there is a bunch of tests to confirm method against Marie's work at REFL, before trying the POP sensing path
'''
# %%
import matplotlib.pyplot as plt
import finesse
import finesse.components as fc
import finesse.detectors as fd
import finesse_ligo
import numpy as np
from finesse.solutions import SimpleSolution
from finesse_ligo.actions import InitialLockLIGO, DARM_RF_to_DC
from finesse.analysis.actions import For, Series, Execute, RunLocks, Noxaxis, SensingMatrixDC

from POP_factory import POPWFSFactory, POPWFS_params

finesse.init_plotting()
# %% Load in the parameter file for LHO O4
# apply POP WFS factory which adds POP WFS path to aligo factory
factory = POPWFSFactory("lho_O4.yaml", POPWFS_params)
factory.options.PR2_substrate = True
factory.options.ASC.add = True
factory.options.ASC.close_AC_loops = False 
factory.options.fake_src_gouy = True

model = factory.make()

model.modes(maxtem=4)  # use at least up to 2nd order modes to include mismatch
model.run(InitialLockLIGO())
model.run(DARM_RF_to_DC())
model.fsig.f = 1
    
# %%
# also include some PRM loss, 37.5 ppm following Marie's choice
model.PRM.set_RTL(T=model.PRM.T, L=37.5e-6)
# also 37.5 ppm of BS loss, again following Marie's choice
model.BS.set_RTL(T=model.BS.T, L=37.5e-6)
# make the ETM loss 30 ppm instead of 60, in other words pushing some loss to the ITM to start
model.ETMX.set_RTL(T=model.ETMX.T, L=30e-6)

# %%
# use an amplitude detector to calculate the reflected field instead of reflected power
model.add(fd.AmplitudeDetector('carrier_amp', model.PRMAR.p2.o, 0))
model.add(fd.AmplitudeDetector('REFLRF9_amp', model.PRMAR.p2.o, model.f1.value))
model.add(fd.AmplitudeDetector('REFLRF45_amp', model.PRMAR.p2.o, model.f2.value))


# get the gouy phase of the POP ASC detector
model.add(fd.gouy.Gouy(name='pop_ASC_gouy_y', from_node=model.PR3.p1.o, to_node=model.pop_ASC.p1, direction='y'))
model.add(fd.gouy.Gouy(name='pop_ASC_gouy_x', from_node=model.PR3.p1.o, to_node=model.pop_ASC.p1, direction='x'))

# %%
# first imitate Marie's model, add loss to the ITMs between 10 and 1000 ppm
n=100
loss = np.zeros(n)
refldc = np.zeros(n, dtype=np.complex128)
refl9 = np.zeros(n, dtype=np.complex128)
refl45 = np.zeros(n, dtype=np.complex128)
prg = np.zeros(n)
for i, l in enumerate(np.geomspace(10e-6, 1000e-6, n)):
    with model.temporary_parameters():
        model.ITMX.set_RTL(T=0.015, L=l)
        DC = model.run()
        loss[i] = l/1e-6
        refldc[i] = DC['carrier_amp']
        refl9[i] = DC['REFLRF9_amp']
        refl45[i] = DC['REFLRF45_amp']
        prg[i] = DC['PRG']
# %%
plt.figure()
plt.semilogx(loss, np.abs(refldc), label = 'carrier')
plt.semilogx(loss, np.abs(refl9), label = '9 MHz')
plt.semilogx(loss, np.abs(refl45), label = '45 MHz')
plt.xlabel('ITMX loss [ppm]')
plt.ylabel('Reflected Field [W]')
plt.title('Fields at REFL Port')
plt.legend()

# %%
fig, ax1 = plt.subplots()
color = 'blue'
ax1.semilogx(loss, np.abs(refldc), label = 'carrier', color=color, linestyle='-')
ax1.semilogx(loss, np.abs(refl9), label = '9 MHz', color=color, linestyle='--')
ax1.semilogx(loss, np.abs(refl45), label = '45 MHz', color=color, linestyle='-.')

ax1.set_xlabel('ITMX loss [ppm]')
ax1.set_ylabel('Reflected Field [W]', color=color)
ax1.set_ylim(0,5)
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper right')

ax2 = ax1.twinx()

color = 'red'
ax2.semilogx(loss, prg, label = 'PRG', color=color)

ax2.set_ylabel('Power recycling gain [W/W]', color=color)
ax2.set_ylim(10, 60)
ax2.tick_params(axis='y', labelcolor=color)
#plt.title('Fields at REFL Port')

# %%
# set DoFs
dofs_P = ('CHARD_P', 'PRC2_P', 'INP1_P')
dofs_Y = ('CHARD_Y', 'PRC2_Y', 'INP1_Y')

readouts_P = ('REFL_A_WFS45y', 'REFL_B_WFS45y', 
              'REFL_A_WFS9y', 'REFL_B_WFS9y')

readouts_Y = ('REFL_A_WFS45x', 'REFL_B_WFS45x', 
              'REFL_A_WFS9x', 'REFL_B_WFS9x')


# %%
n = 50
loss = np.zeros(n)
refldc = np.zeros(n, dtype=np.complex128)
refl9 = np.zeros(n, dtype=np.complex128)
refl45 = np.zeros(n, dtype=np.complex128)
prg = np.zeros(n)
loss = np.zeros(n)
sensing_P = np.zeros((n, 3, 4), dtype=np.complex128)
sensing_Y = np.zeros((n, 3, 4), dtype=np.complex128)
for i, l in enumerate(np.geomspace(10e-6, 1000e-6, n)):
    with model.temporary_parameters():
        model.ITMX.set_RTL(T=0.015, L=l)
        sol = model.run(
            Series(SensingMatrixDC(
                dofs_P, readouts_P, d_dof=1e-10, name = 'pitch'),
                SensingMatrixDC(
                dofs_Y, readouts_Y, d_dof=1e-10, name = 'yaw'),
            Noxaxis(name='dc_power'))
                )
        refldc[i] = sol['dc_power']['carrier_amp']
        refl9[i] = sol['dc_power']['REFLRF9_amp']
        refl45[i] = sol['dc_power']['REFLRF45_amp']
        prg[i] = sol['dc_power']['PRG']
        sensing_P[i, :, :] = sol['pitch'].out
        sensing_Y[i, :, :] = sol['yaw'].out
    loss[i] = l/1e-6
# %%
for i in range(len(readouts_P)):
    fig, ax = plt.subplots(2,1, figsize=(10,8))
    for j in range(len(dofs_P)):
        ax[0].loglog(loss, np.abs(sensing_P[:, j, i]), label=dofs_P[j])
        ax[1].semilogx(loss, np.angle(sensing_P[:, j, i], deg=True))
    ax[0].set_title(readouts_P[i])
    ax[0].legend()
    ax[1].set_xlabel('ITMX loss [ppm]')
    ax[0].set_ylabel('Sensing Gain [W/rad]')
    ax[1].set_ylabel('Sensing Phase [deg]')

for i in range(len(readouts_Y)):
    fig, ax = plt.subplots(2,1, figsize=(10,8))
    for j in range(len(dofs_Y)):
        ax[0].loglog(loss, np.abs(sensing_Y[:, j, i]), label=dofs_Y[j])
        ax[1].semilogx(loss, np.angle(sensing_Y[:, j, i], deg=True))
    ax[0].set_title(readouts_Y[i])
    ax[0].legend()
    ax[1].set_xlabel('ITMX loss [ppm]')
    ax[0].set_ylabel('Sensing Gain [W/rad]')
    ax[1].set_ylabel('Sensing Phase [deg]')
# %%
# reasonable results at REFL port, let's test the POP WFS

dofs_P = ['PRC2_P']
dofs_Y = ['PRC2_Y']

readouts_P = ['ASC_POP_A_36y', 'ASC_POP_A_45y']
readouts_Y = ['ASC_POP_A_36x', 'ASC_POP_A_45x']

n = 50
loss = np.zeros(n)
prg = np.zeros(n)
loss = np.zeros(n)
sensing_P = np.zeros((n, 1, 2), dtype=np.complex128)
sensing_Y = np.zeros((n, 1, 2), dtype=np.complex128)
for i, l in enumerate(np.geomspace(10e-6, 1000e-6, n)):
    with model.temporary_parameters():
        model.ITMX.set_RTL(T=0.015, L=l)
        sol = model.run(
            Series(SensingMatrixDC(
                dofs_P, readouts_P, d_dof=1e-10, name = 'pitch'),
                SensingMatrixDC(
                dofs_Y, readouts_Y, d_dof=1e-10, name = 'yaw'),
            Noxaxis(name='dc_power'))
                )
        prg[i] = sol['dc_power']['PRG']
        sensing_P[i, :, :] = sol['pitch'].out
        sensing_Y[i, :, :] = sol['yaw'].out
    loss[i] = l/1e-6
# %%

for i in range(len(readouts_P)):
    fig, ax = plt.subplots(2,1, figsize=(10,8))
    for j in range(len(dofs_P)):
        ax[0].semilogx(loss, np.abs(sensing_P[:, j, i]), label= dofs_P[j])
        ax[1].semilogx(loss, np.angle(sensing_P[:, j, i], deg=True))
    ax[0].set_title(readouts_P[i])
    ax[0].legend()
    ax[1].set_xlabel('ITMX loss [ppm]')
    ax[0].set_ylabel('Sensing Gain [W/rad]')
    ax[1].set_ylabel('Sensing Phase [deg]')
    ax[1].set_ylim(-180, 180)

for i in range(len(readouts_Y)):
    fig, ax = plt.subplots(2,1, figsize=(10,8))
    for j in range(len(dofs_Y)):
        ax[0].semilogx(loss, np.abs(sensing_Y[:, j, i]), label = dofs_Y[j])
        ax[1].semilogx(loss, np.angle(sensing_Y[:, j, i], deg=True))
    ax[0].set_title(readouts_Y[i])
    ax[0].legend()
    ax[1].set_xlabel('ITMX loss [ppm]')
    ax[0].set_ylabel('Sensing Gain [W/rad]')
    ax[1].set_ylabel('Sensing Phase [deg]')
    ax[1].set_ylim(-180, 180)

# %%
# change distance of POP WFS location
# WIP below here
n = 5
length = np.zeros(n)
gouy_x = np.zeros(n)
gouy_y = np.zeros(n)
loss = np.zeros(n)



for i, l in enumerate(np.geomspace(10e-6, 1000e-6, n)):
    for j, d in enumerate(np.linspace(0, 0.3, n)):    
        with model.temporary_parameters():
            model.BS_ASC.L = d

            model.ITMX.set_RTL(T=0.015, L=l)
            sol = model.run(
                Series(
                    SensingMatrixDC(dofs_P, readouts_P, d_dof=1e-10, name = 'pitch'),
                    SensingMatrixDC(dofs_Y, readouts_Y, d_dof=1e-10, name = 'yaw'),
                    Noxaxis(name='dc_power')))

            sol = model.run()
            sensing_P[i, j, :, :] = sol['pitch'].out
            sensing_Y[i, j, :, :] = sol['yaw'].out

            
        gouy_x[j] = sol['pop_ASC_gouy_x']
        gouy_y[j] = sol['pop_ASC_gouy_y']
        length[j] = d
    loss[i] = l
# %%
