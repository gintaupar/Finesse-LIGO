#########################################################################
## Modeling of ZM2, ZM4, and ZM5 PSAMS and propgation of squeezed beam through VIP
#########################################################################

## Based on beam profiling from T2200151
# PSAM scan results were recalculated using a ZM4 to ZM5 distance of 1.69125 m

import numpy as np
import matplotlib.pyplot as plt
import finesse
from finesse_ligo.factory import aligo
import pandas as pd

finesse.init_plotting()
factory = aligo.ALIGOFactory("yaml\\llo_O4_addVIP.yaml")
factory.options.SQZ.add = True
factory.options.SQZ.add_VIP = True
factory.options.SQZ.OPO_wedges = False
factory.options.SQZ.set_opo_cavity = True

model = factory.make()

VIP_ZM1 = model.VIP_ZM1.L
ZM1_ZM2 = model.ZM1_ZM2.L


# Conversion between strain voltage and ZM2 ROCX and ROCY
def zm2_roc(strain, uncertainty=False):
    dmx = 0.9
    dmy = 2.0
    dbx = 0.9
    dby = 1.9
    mx = -39.2  # mD/v
    my = -38.5  # mD/v
    bx = 2305.1  # mD
    by = 2296.3  # mD

    rocx = 2 / ((mx * strain + bx) * 1e-3)
    rocy = 2 / ((my * strain + by) * 1e-3)
    if uncertainty:
        return (rocx, rocy, (2 * np.sqrt(dbx ** 2 + dmx ** 2 * strain ** 2) / (bx + mx * strain) ** 2) * 1e3,
                (2 * np.sqrt(dby ** 2 + dmy ** 2 * strain ** 2) / (by + my * strain) ** 2) * 1e3)
    else:
        return (rocx, rocy)


# Conversion between strain voltage and ZM4 ROCX and ROCY
def zm4_roc(strain, uncertainty=False):
    dmx = 1.7
    dmy = 1.2
    dbx = 2.6
    dby = 1.8
    mx = -29.6  # mD/v
    my = -29.9  # mD/v
    bx = -105.5  # mD
    by = -156.6  # mD

    rocx = 2 / ((mx * strain + bx) * 1e-3)
    rocy = 2 / ((my * strain + by) * 1e-3)
    if uncertainty:
        return (rocx, rocy, (2 * np.sqrt(dbx ** 2 + dmx ** 2 * strain ** 2) / (bx + mx * strain) ** 2) * 1e3,
                (2 * np.sqrt(dby ** 2 + dmy ** 2 * strain ** 2) / (by + my * strain) ** 2) * 1e3)
    else:
        return (rocx, rocy)


# Conversion between strain voltage and ZM5 ROCX and ROCY
def zm5_roc(strain, uncertainty=False):
    dmx = 0.8
    dmy = 1.1
    dbx = 0.6
    dby = 0.9
    mx = -34.2  # mD/v
    my = -38.4  # mD/v
    bx = 595.6  # mD
    by = 615.2  # mD

    rocx = 2 / ((mx * strain + bx) * 1e-3)
    rocy = 2 / ((my * strain + by) * 1e-3)
    if uncertainty:
        return (rocx, rocy, (2 * np.sqrt(dbx ** 2 + dmx ** 2 * strain ** 2) / (bx + mx * strain) ** 2) * 1e3,
                (2 * np.sqrt(dby ** 2 + dmy ** 2 * strain ** 2) / (by + my * strain) ** 2) * 1e3)
    else:
        return (rocx, rocy)


# Mode mismatch between two beams with complex beam parameters q1x, q1y and q2x, q2y
def mode_mm(q1x, q1y, q2x, q2y):
    mmx = 1 - finesse.gaussian.BeamParam.mismatch(finesse.gaussian.BeamParam(q=q1x), finesse.gaussian.BeamParam(q=q2x))
    mmy = 1 - finesse.gaussian.BeamParam.mismatch(finesse.gaussian.BeamParam(q=q1y), finesse.gaussian.BeamParam(q=q2y))
    mm2 = np.sqrt(mmx * mmy)
    return (1 - mm2) * 100


def plot(ps, ax, offset=0, ls='-', reverse=1, color='blue', label=False):
    data = ps.all_segments()
    for key in data.keys():
        if label is None:
            ax.plot(reverse * data[key][0] + offset, data[key][1]['beamsize'] * 1e3, linestyle=ls, color=color,
                    label=key.name)
        else:
            ax.plot(reverse * data[key][0] + offset, data[key][1]['beamsize'] * 1e3, linestyle=ls, color=color,
                    label=label)
            label = "_nolegend_"


def fliprel(q):
    return (-1 * np.conjugate(q))


## Comparison between ZM4, ZM5 model and measurements
fig, ax = plt.subplots(2, 1, figsize=(7, 5))
fig.tight_layout(pad=3.0)

model.ZM4.Rcx = zm4_roc(-.79)[0]
model.ZM4.Rcy = zm4_roc(-.79)[1]
model.ZM5.Rcx = zm5_roc(0)[0]
model.ZM5.Rcy = zm5_roc(0)[1]

psx = model.propagate_beam(from_node=factory.VIP_OUT_B, to_node=model.ZM6.p1,
                           q_in=finesse.gaussian.BeamParam(w0=(630.54) * 1e-6, z=-(model.VIP_ZM4.L + (-1.61))),
                           direction='x')
psy = model.propagate_beam(from_node=factory.VIP_OUT_B, to_node=model.ZM6.p1,
                           q_in=finesse.gaussian.BeamParam(w0=(674.65) * 1e-6, z=-(model.VIP_ZM4.L + (-1.33))),
                           direction='y')

plot(psx, ax[0], color='red')
ax[0].set_xlabel('Distance from M4 (m)')
ax[0].set_ylabel('Beam size (mm)')
ax[0].set_title('Horizontal Beam Propagation')
ax[0].axvline(x=psx.positions[model.ZM4], ls='--', color='black')
ax[0].axvline(x=psx.positions[model.ZM5], ls='--', color='black')
ax[0].text(s='ZM4', x=psx.positions[model.ZM4], y=.6)
ax[0].text(s='ZM5', x=psx.positions[model.ZM5], y=.6)

plot(psy, ax[1], color='blue')
ax[1].set_xlabel('Distance from M4 (m)')
ax[1].set_ylabel('Beam size (mm)')
ax[1].set_title('Vertical Beam Propagation')
ax[1].axvline(x=psx.positions[model.ZM4], ls='--', color='black')
ax[1].axvline(x=psx.positions[model.ZM5], ls='--', color='black')
ax[1].text(s='ZM4', x=psx.positions[model.ZM4], y=.6)
ax[1].text(s='ZM5', x=psx.positions[model.ZM5], y=.6)

# Beam measurements before ZM4
pp_ZM4 = np.array([-.393, -.336, -.157, 0]) + psx.positions[model.ZM4]
wx_ZM4 = np.array([1810, 1890, 2000, 2130]) * 1e-3 / 2
wy_ZM4 = np.array([1650, 1680, 1800, 1890]) * 1e-3 / 2

# Beam measurements before ZM5
pp_ZM5 = np.array([-.400, -.297, -.137, 0]) + psx.positions[model.ZM5]
wx_ZM5 = np.array([3510, 3670, 3710, 3970]) * 1e-3 / 2
wy_ZM5 = np.array([3240, 3370, 3520, 3725]) * 1e-3 / 2

ax[0].plot(pp_ZM4, wx_ZM4, '*', color='red')
ax[1].plot(pp_ZM4, wy_ZM4, '*', color='blue')

ax[0].plot(pp_ZM5, wx_ZM5, '*', color='red')
ax[1].plot(pp_ZM5, wy_ZM5, '*', color='blue')

######################################
## Beam Propagation to the OMC
######################################

zm4_rc = zm4_roc(1.0)
zm5_rc = zm5_roc(1.0)
model.ZM4.Rcx = zm4_rc[0]
model.ZM4.Rcy = zm4_rc[1]
model.ZM5.Rcx = zm5_rc[0]
model.ZM5.Rcy = zm5_rc[1]

psx = model.propagate_beam(from_node=factory.VIP_OUT_B, to_node=model.OMC_IC.p2.o,
                           q_in=finesse.gaussian.BeamParam(w0=(630.54) * 1e-6, z=-(model.VIP_ZM4.L + (-1.61))),
                           direction='x', via_node=model.SRM.p2.o)
psy = model.propagate_beam(from_node=factory.VIP_OUT_B, to_node=model.OMC_IC.p2.o,
                           q_in=finesse.gaussian.BeamParam(w0=(674.65) * 1e-6, z=-(model.VIP_ZM4.L + (-1.33))),
                           direction='y', via_node=model.SRM.p2.o)

fig, ax = plt.subplots()
plot(psx, ax, color='red', label='Horizontal')
plot(psy, ax, color='blue', label='Vertical')
ax.set_xlabel('Distance from M4 (m)')
ax.set_ylabel('Beam Size (mm)')
print(
    f'Mode Mismatch to OMC: {mode_mm(model.cavOMC.q[0].q, model.cavOMC.q[1].q, psx.qs[model.OMC_IC.p2.o].q, psy.qs[model.OMC_IC.p2.o].q)}')

ax.axvline(x=psx.positions[model.ZM4], color='black', ls='--')
ax.axvline(x=psx.positions[model.ZM5], color='black', ls='--')
ax.axvline(x=psx.positions[model.ZM6], color='black', ls='--')
ax.axvline(x=psx.positions[model.OFI], color='black', ls='--')
ax.axvline(x=psx.positions[model.SRM], color='black', ls='--')
ax.axvline(x=psx.positions[model.OM1], color='black', ls='--')
ax.axvline(x=psx.positions[model.OM2], color='black', ls='--')
ax.axvline(x=psx.positions[model.OM3], color='black', ls='--')

off = -.5
size = 7
ax.text(x=psx.positions[model.ZM4] + off, y=2.30, s='ZM4', size=size)
ax.text(x=psx.positions[model.ZM5] + off, y=2.30, s='ZM5', size=size)
ax.text(x=psx.positions[model.ZM6] + off, y=2.30, s='ZM6', size=size)
ax.text(x=psx.positions[model.OFI] + off, y=2.30, s='OFI', size=size)
ax.text(x=psx.positions[model.SRM] + off, y=2.30, s='SRM', size=size)
ax.text(x=psx.positions[model.OM1] + 2 * off, y=2.30, s='OM1', size=size)
ax.text(x=psx.positions[model.OM2] + 2 * off, y=2.30, s='OM2', size=size)
ax.text(x=psx.positions[model.OM3] + off, y=2.30, s='OM3', size=size)
ax.legend(loc='upper left', fontsize=7)

plt.show()

#####################################
## Mode matching to the OMC
#####################################

# ZM4, ZM5 mode matching map
n = 25
zm4_strain = np.linspace(0, 2, n)
zm5_strain = np.linspace(0, 2, n)
mm = []
for i in range(n):
    rcx1, rcy1 = zm4_roc(zm4_strain[i])
    model.ZM4.Rcx = rcx1
    model.ZM4.Rcy = rcy1
    for j in range(n):
        rcx2, rcy2 = zm5_roc(zm5_strain[j])
        model.ZM5.Rcx = rcx2
        model.ZM5.Rcy = rcy2

        psx = model.propagate_beam(from_node=factory.VIP_OUT_B, to_node=model.OMC_IC.p2.o,
                                   q_in=finesse.gaussian.BeamParam(w0=(630.54) * 1e-6, z=-(model.VIP_ZM4.L + (-1.61))),
                                   direction='x', via_node=model.SRM.p2.o)

        psy = model.propagate_beam(from_node=factory.VIP_OUT_B, to_node=model.OMC_IC.p2.o,
                                   q_in=finesse.gaussian.BeamParam(w0=(674.65) * 1e-6, z=-(model.VIP_ZM4.L + (-1.33))),
                                   direction='y', via_node=model.SRM.p2.o)

        mismatch = mode_mm(model.cavOMC.q[0].q, model.cavOMC.q[1].q, psx.qs[model.OMC_IC.p2.o].q,
                           psy.qs[model.OMC_IC.p2.o].q)

        mm.append([zm4_strain[i], zm5_strain[j], mismatch])

d = pd.DataFrame(mm, columns=['Zm4', 'Zm5', 'mismatch'])
data = d.pivot(index='Zm4', columns='Zm5', values='mismatch')
fig, ax = plt.subplots()
pc = ax.pcolormesh(data.index, data.columns, data.values, cmap='magma', alpha=1, antialiased=True)
pc.set_edgecolor('face')
C = ax.contour(data.index, data.columns, data.values, [1.5, 2, 3, 4, 5], cmap='Pastel1')
ax.clabel(C, C.levels, inline=True, fontsize=10, fmt=lambda x: f'{x}%')
ax.set_xlabel('ZM5 Strain (V)')
ax.set_ylabel('ZM4 Strain (V)')
ax.plot(1, 1, '*', color='white')
plt.show()

####################
## Modeling ZM2
####################
fig, ax = plt.subplots(2, 1, figsize=(7, 5))
fig.tight_layout(pad=3.0)
psx = model.propagate_beam(from_node=factory.VIP_OUT_A, to_node=model.FC1.p2.i,
                           q_in=finesse.gaussian.BeamParam(w0=(211.5) * 1e-6,
                                                           z=-((model.VIP_ZM1.L + model.ZM1_ZM2.L) - (1.509))),
                           direction='x')
psy = model.propagate_beam(from_node=factory.VIP_OUT_A, to_node=model.FC1.p2.i,
                           q_in=finesse.gaussian.BeamParam(w0=(212.817635) * 1e-6,
                                                           z=-((model.VIP_ZM1.L + model.ZM1_ZM2.L) - (1.513))),
                           direction='y')

plot(psx, ax[0], color='red')
ax[0].set_xlabel('Distance from M3 (m)')
ax[0].set_ylabel('Beam size (mm)')
ax[0].set_title('Horizontal Beam Propagation')
ax[0].axvline(x=psx.positions[model.ZM2], ls='--', color='black')
ax[0].axvline(x=psx.positions[model.ZM3], ls='--', color='black')
ax[0].text(s='ZM2', x=psx.positions[model.ZM2], y=.6)
ax[0].text(s='ZM3', x=psx.positions[model.ZM3], y=.6)

plot(psy, ax[1], color='blue')
ax[1].set_xlabel('Distance from M3 (m)')
ax[1].set_ylabel('Beam size (mm)')
ax[1].set_title('Vertical Beam Propagation')
ax[1].axvline(x=psx.positions[model.ZM2], ls='--', color='black')
ax[1].axvline(x=psx.positions[model.ZM3], ls='--', color='black')
ax[1].text(s='ZM2', x=psx.positions[model.ZM2], y=.6)
ax[1].text(s='ZM3', x=psx.positions[model.ZM3], y=.6)

# Beam Measurements before ZM2
pp_ZM2 = np.array([-825, -775, -725, -625, -525, -375, -275, -175, 0]) / 1000 + psx.positions[model.ZM2]
wx_ZM2 = np.array([2210, 2385, 2510, 2833, 3126, 3828, 4055, 4305, 4839]) * 1e-3 / 2
wy_ZM2 = np.array([2207, 2360, 2543, 2838, 3140, 3777, 4041, 4295, 4815]) * 1e-3 / 2

ax[0].plot(pp_ZM2, wx_ZM2, '*', color='red')
ax[1].plot(pp_ZM2, wy_ZM2, '*', color='blue')
plt.show()

#############################################
## Mode Mismatch to the filter cavity
#############################################

model.ZM2_ZM3.L = 1.805  # Estimation from T2200151
model.ZM2.Rcx = zm2_roc(-2.3)[0]
model.ZM2.Rcy = zm2_roc(-2.3)[1]

ps_fc = model.propagate_beam(from_node=model.FC1AR.p2.i, to_node=model.FC1.p2)
fc_q = ps_fc.qs[model.FC1AR.p2.i].q  # Filter cavity q at FC1AR Surface

psx = model.propagate_beam(from_node=factory.VIP_OUT_A, to_node=model.FC1AR.p2.i,
                           q_in=finesse.gaussian.BeamParam(w0=(211.5) * 1e-6, z=-((VIP_ZM1 + ZM1_ZM2) - (1.509))),
                           direction='x')
psy = model.propagate_beam(from_node=factory.VIP_OUT_A, to_node=model.FC1AR.p2.i,
                           q_in=finesse.gaussian.BeamParam(w0=(212.817635) * 1e-6, z=-((VIP_ZM1 + ZM1_ZM2) - (1.513))),
                           direction='y')
mismatch = mode_mm(psx.qs[model.FC1AR.p2.i].q, psy.qs[model.FC1AR.p2.i].q, fc_q, fc_q)
print(f'Mismatch to filter cavity {mismatch}')

m = []
stn = np.linspace(0, -3, 500)
for s in stn:
    model.ZM2.Rcx = zm2_roc(s)[0]
    model.ZM2.Rcy = zm2_roc(s)[1]
    psx = model.propagate_beam(from_node=factory.VIP_OUT_A, to_node=model.FC1AR.p2.i,
                               q_in=finesse.gaussian.BeamParam(w0=(211.5) * 1e-6, z=-((VIP_ZM1 + ZM1_ZM2) - (1.509))),
                               direction='x')
    psy = model.propagate_beam(from_node=factory.VIP_OUT_A, to_node=model.FC1AR.p2.i,
                               q_in=finesse.gaussian.BeamParam(w0=(212.817635) * 1e-6,
                                                               z=-((VIP_ZM1 + ZM1_ZM2) - (1.513))), direction='y')
    mm = mode_mm(psx.qs[model.FC1AR.p2.i].q, psy.qs[model.FC1AR.p2.i].q, fc_q, fc_q)
    m.append(mm)
plt.plot(stn, m)
plt.xlabel('ZM2 Strain (V)')
plt.ylabel('Mode Mismatch (%)')

#######################################
## VIP Path
#######################################

model.ZM2_ZM3.L = 1.805 # Estimated distance from T2200151
model.ZM2.Rcx = zm2_roc(-2.7)[0]

ps_opo = model.propagate_beam(from_node=factory.CLF_IN_PORT.i, to_node=model.FC1.p2)
ps_meas = model.propagate_beam(from_node=model.VIP_M3.p2, to_node=model.FC1.p2,
                               q_in=finesse.BeamParam(w0=(211.5) * 1e-6,
                                                      z=-((model.VIP_ZM1.L + model.ZM1_ZM2.L) - (1.509))))

fig, ax = plt.subplots()
plot(ps_opo, ax, label='OPO Mode', color='#41a4ba')
plot(ps_meas, ax, offset=ps_opo.positions[model.VIP_M3], ls='--', label='Measured Beam on ZM2')
ax.axvline(x=ps_opo.positions[model.ZM1], color='black', ls='--')
ax.axvline(x=ps_opo.positions[model.ZM2], color='black', ls='--')
ax.axvline(x=ps_opo.positions[model.ZM3], color='black', ls='--')
ax.axvline(x=ps_opo.positions[model.FC1], color='black', ls='--')

off = -.5
size = 7
ax.text(x=ps_opo.positions[model.ZM1] + off, y=11, s='ZM1', color='black', size=size)
ax.text(x=ps_opo.positions[model.ZM2] + off, y=11, s='ZM2', color='black', size=size)
ax.text(x=ps_opo.positions[model.ZM3] + off, y=11, s='ZM3', color='black', size=size)
ax.text(x=ps_opo.positions[model.FC1] + off, y=11, s='FC1', color='black', size=size)
mm = []
ax.set_xlabel('Distance from CLF_IN_PORT (m)')
ax.set_ylabel('Beam Size (mm)')
ax.legend()
print(
    f'Mismatch between OPO mode and ZM2 measurement: {mode_mm(ps_opo.qs[model.ZM2.p1.i].q, ps_opo.qs[model.ZM2.p1.i].q, ps_meas.qs[model.ZM2.p1.i].q, ps_meas.qs[model.ZM2.p1.i].q)}')

ps_opo = model.propagate_beam(from_node=factory.CLF_IN_PORT.i, to_node=model.ZM4.p1, via_node=model.FC1.p2.o)
ps_meas = model.propagate_beam(from_node=model.VIP_M3.p2, to_node=model.ZM4.p1,
                               q_in=finesse.BeamParam(w0=(211.5) * 1e-6,
                                                      z=-((model.VIP_ZM1.L + model.ZM1_ZM2.L) - (1.509))),
                               via_node=model.FC1.p2.o)
ps_distance = model.propagate_beam(from_node=model.FC1.p2.o, to_node=model.ZM4.p1)

zm4_q = finesse.BeamParam(w0=630e-6, z=-1.615)
ps_meas_zm4 = model.propagate_beam(from_node=model.ZM4.p1, to_node=model.ZM2.p1, q_in=zm4_q,
                                   reverse_propagate=True)  # Measured Beam parameter on ZM4

fig, ax = plt.subplots()
plot(ps_opo, ax, label='OPO Mode', color='#41a4ba')
plot(ps_meas, ax, offset=ps_opo.positions[model.VIP_M3], ls='--', label='Measured Beam on ZM2')
plot(ps_meas_zm4, ax, offset=ps_opo.positions[model.ZM4], ls=':', reverse=-1, color='red',
     label='Measured Beam on ZM4')
ax.set_xlim([6, 13])
ax.set_ylim([0, 4])
ax.axvline(x=ps_opo.positions[model.FC1] + ps_distance.positions[model.ZM2], ls='--', color='black')
ax.axvline(x=ps_opo.positions[model.FC1] + ps_distance.positions[model.VIP_M3], ls='--', color='black')
ax.axvline(x=ps_opo.positions[model.FC1] + ps_distance.positions[model.VIP_M4], ls='--', color='black')

off = -.5
size = 7
ax.text(x=ps_opo.positions[model.FC1] + ps_distance.positions[model.ZM2] + off, y=4.1, s='ZM2', size=size)
ax.text(x=ps_opo.positions[model.FC1] + ps_distance.positions[model.VIP_M3] + off, y=4.10, s='M3', size=size)
ax.text(x=ps_opo.positions[model.FC1] + ps_distance.positions[model.VIP_M4] + off, y=4.10, s='M4', size=size)

print(
    f'Mismatch between ZM2 measurement and ZM4 measurement: {mode_mm(ps_meas.qs[model.ZM4.p1.i].q, ps_meas.qs[model.ZM4.p1.i].q, fliprel(zm4_q), fliprel(zm4_q))}')
ax.legend()
ax.set_xlabel('Distance from CLF_IN_PORT (m)')
ax.set_ylabel('Beam Size (mm)')
plt.show()
