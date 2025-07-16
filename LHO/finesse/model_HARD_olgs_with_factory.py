# %%
import finesse.ligo
import numpy as np
import matplotlib.pyplot as plt
from finesse.plotting import bode
from finesse.ligo.suspension import QUADSuspension
from finesse.ligo.ASC_controllers import get_controller
from finesse.analysis.actions import (
    FrequencyResponse,
    Series,
)
from finesse.ligo.factory import aligo
import sys
from finesse.plotting import bode
import finesse.analysis.actions as fac
from finesse.ligo.actions import InitialLockLIGO, DARM_RF_to_DC
from finesse.ligo.QUAD_control import get_locking_filters

finesse.init_plotting(fmts=['png'])

# %%
factory = aligo.ALIGOFactory(finesse.ligo.git_path() / "LHO" / "yaml" / "lho_O4.yaml")

factory.options.QUAD_suspension_model = QUADSuspension
factory.options.ASC.add = True
# Leave the ASC loops open, this breaks the loop between the readouts and the controller
factory.options.ASC.close_AC_loops = False 
factory.ASC_input_matrix.DHARD_P['AS_A_WFS45y.Q'] = 1
factory.ASC_input_matrix.CHARD_P['REFL_B_WFS9y.I'] = 1
factory.ASC_input_matrix.CHARD_P['REFL_B_WFS45y.I'] = 1

factory.ASC_input_matrix.DHARD_Y['AS_A_WFS45x.Q'] = 1
factory.ASC_input_matrix.CHARD_Y['REFL_B_WFS9x.I'] = 1
factory.ASC_input_matrix.CHARD_Y['REFL_B_WFS45x.I'] = 1

for dof in ['DHARD_P', 'CHARD_P', 'DHARD_Y', 'CHARD_Y']:
    factory.ASC_controller[dof] = (
        dof+"_cntrl",
        *get_controller(dof.lower()),
    )
    
for mirror in ["ITMX", "ITMY", "ETMX", "ETMY"]:
    factory.local_drives["Y"][mirror].clear()
    factory.local_drives["Y"][mirror][f"{mirror}_sus.L2.F_yaw"] = ('L2Y_'+mirror+'_cntrl', 
                                                                   np.concatenate((get_locking_filters('L2', mirror, 'Y')[0], get_locking_filters('L3', mirror, 'Y')[0])),
                                                                   np.concatenate((get_locking_filters('L2', mirror, 'Y')[1], get_locking_filters('L3', mirror, 'Y')[1])),
                                                                   get_locking_filters('L2', mirror, 'Y')[2]*get_locking_filters('L3', mirror, 'Y')[2])
    factory.local_drives["Y"][mirror][f"{mirror}_sus.M0.F_yaw"] = ('M0Y_'+mirror+'_cntrl', 
                                                                   np.concatenate((get_locking_filters('L2', mirror, 'Y')[0], get_locking_filters('L3', mirror, 'Y')[0], get_locking_filters('M0', mirror, 'Y')[0])),
                                                                   np.concatenate((get_locking_filters('L2', mirror, 'Y')[1], get_locking_filters('L3', mirror, 'Y')[1], get_locking_filters('M0', mirror, 'Y')[1])),
                                                                   get_locking_filters('L2', mirror, 'Y')[2]*get_locking_filters('L3', mirror, 'Y')[2]*get_locking_filters('M0', mirror, 'Y')[2])
    factory.local_drives["P"][mirror].clear()
    factory.local_drives["P"][mirror][f"{mirror}_sus.L2.F_pitch"] = ('L2P_'+mirror+'_cntrl', 
                                                                   np.concatenate((get_locking_filters('L2', mirror, 'P')[0], get_locking_filters('L3', mirror, 'P')[0])),
                                                                   np.concatenate((get_locking_filters('L2', mirror, 'P')[1], get_locking_filters('L3', mirror, 'P')[1])),
                                                                   get_locking_filters('L2', mirror, 'P')[2]*get_locking_filters('L3', mirror, 'P')[2])
    factory.local_drives["P"][mirror][f"{mirror}_sus.M0.F_pitch"] = ('M0P_'+mirror+'_cntrl', 
                                                                   np.concatenate((get_locking_filters('L2', mirror, 'P')[0], get_locking_filters('L3', mirror, 'P')[0], get_locking_filters('M0', mirror, 'P')[0])),
                                                                   np.concatenate((get_locking_filters('L2', mirror, 'P')[1], get_locking_filters('L3', mirror, 'P')[1], get_locking_filters('M0', mirror, 'P')[1])),
                                                                   get_locking_filters('L2', mirror, 'P')[2]*get_locking_filters('L3', mirror, 'P')[2]*get_locking_filters('M0', mirror, 'P')[2])

# %%
P_target = 360e3
model = factory.make()
# radius of curvature change at high power
ITM_Rc_D_per_W = -46e-6
ETM_Rc_D_per_W = -33.46e-6
model.ITMXlens.f = model.ITMXlens.f
model.ITMYlens.f = model.ITMYlens.f
model.ITMX.Rc = 2 / (2 / model.ITMX.Rc + ITM_Rc_D_per_W * P_target * 0.5e-6)
model.ITMY.Rc = 2 / (2 / model.ITMY.Rc + ITM_Rc_D_per_W * P_target * 0.5e-6)
model.ETMX.Rc = 2 / (2 / model.ETMX.Rc + ETM_Rc_D_per_W * P_target * 0.5e-6 * 3 / 5)
model.ETMY.Rc = 2 / (2 / model.ETMY.Rc + ETM_Rc_D_per_W * P_target * 0.5e-6 * 3 / 5)

model.modes(maxtem=4)
sol = model.run(fac.Series(InitialLockLIGO(), DARM_RF_to_DC()))

_out = model.run()
# Rescale to correct power
model.L0.P *= P_target /((_out['Px']+_out['Px'])/2)
out = model.run()
assert np.allclose(out['Px'], P_target, rtol=1e-3)

model.DHARD_P_cntrl.k *= 5
model.DHARD_Y_cntrl.k *= 4.9
model.CHARD_P_cntrl.k *= -2.4
model.CHARD_Y_cntrl.k *= -8

DHARD_P = model.DHARD_P
DHARD_Y = model.DHARD_Y
CHARD_P = model.CHARD_P
CHARD_Y = model.CHARD_Y

# %%
# Plot what the DHARD_P injection is going into 
model.display_signal_blockdiagram("DHARD_P.AC.i")

# %%
# sanity check of free sus plants at target power
F_Hz = np.geomspace(0.1, 10, 200)

sol1 = model.run(
    FrequencyResponse(
        F_Hz,
        [DHARD_P.AC.i, CHARD_P.AC.i, DHARD_Y.AC.i, CHARD_Y.AC.i],
        [DHARD_P.AC.o, CHARD_P.AC.o, DHARD_Y.AC.o, CHARD_Y.AC.o],
    )
)

bode(sol1.f, sol1["DHARD_P.AC.i", "DHARD_P.AC.o"], label="DHARD_P", wrap=True)
bode(sol1.f, sol1["CHARD_P.AC.i", "CHARD_P.AC.o"], label="CHARD_P", wrap=True)
bode(sol1.f, sol1["DHARD_Y.AC.i", "DHARD_Y.AC.o"], label="DHARD_Y", wrap=True)
bode(sol1.f, sol1["CHARD_Y.AC.i", "CHARD_Y.AC.o"], label="CHARD_Y", wrap=True)

# %%
F_Hz = np.geomspace(0.1, 10, 200)

sol = model.run(
    Series(
        FrequencyResponse(
            F_Hz,
            model.DHARD_P_cntrl.p1,
            DHARD_P.AC.o,
            open_loop=True,
            name="DHARD_P_olg",
        ),
        FrequencyResponse(
            F_Hz,
            model.CHARD_P_cntrl.p1,
            CHARD_P.AC.o,
            open_loop=True,
            name="CHARD_P_olg",
        ),
        FrequencyResponse(
            F_Hz,
            model.DHARD_Y_cntrl.p1,
            DHARD_Y.AC.o,
            open_loop=True,
            name="DHARD_Y_olg",
        ),
        FrequencyResponse(
            F_Hz,
            model.CHARD_Y_cntrl.p1,
            CHARD_Y.AC.o,
            open_loop=True,
            name="CHARD_Y_olg",
        ),
    )
)


# %%
if "pytest" not in sys.modules:
    colors = ("xkcd:burgundy", "xkcd:dark olive", "xkcd:salmon pink", "xkcd:cornflower")
    names = ("DHARD_P_olg", "CHARD_P_olg", "DHARD_Y_olg", "CHARD_Y_olg")
    dofs_in = ("DHARD_P.AC.i", "CHARD_P.AC.i", "DHARD_Y.AC.i", "CHARD_Y.AC.i")
    dofs_out = ("DHARD_P.AC.o", "CHARD_P.AC.o", "DHARD_Y.AC.o", "CHARD_Y.AC.o")
    for j in range(4):
        fig, ax = plt.subplots(nrows=2, ncols=1)
        ax[0].semilogx(
            F_Hz,
            20 * np.log10(np.abs(sol[names[j]][dofs_in[j], dofs_out[j]])),
            color=colors[j],
        )
        ax[1].semilogx(
            F_Hz,
            np.angle(sol[names[j]][dofs_in[j], dofs_out[j]], deg=True),
            color=colors[j],
        )
        ax[0].set_title(dofs_in[j][:7])
        ax[0].set_ylabel("Mag")
        ax[1].set_xlabel("Freq [Hz]")
        ax[1].set_ylabel("Phase [deg]")

# %%
# errorbar functions
def csd_variance_mag(coh, averages):
    """Relative CSD variance along the mag axis
    Variance CSD / Mean CSD^2
    coh in this case is gamma**2, the power coherence.
    Bendat and Piersol Random Data Analysis and Measurement Procedures Eq 9.31
    Copied from C. Cahillane
    """
    return 1 / (averages * coh)

def csd_variance_phase(coh, averages):
    """Relative CSD variance along the minor axis
    Variance CSD / Mean CSD^2
    Copied from C. Cahillane
    """
    return (1 - coh) / (2 * averages * coh)

# %%
dir = 'LHO/Measurements/'
DHP_file = np.loadtxt(dir+'DHARD_P_tf_20230810.txt')
freq_DHP = DHP_file[:,0]
in1_exc_DHP = DHP_file[:,1] + 1j*DHP_file[:,2]
in2_exc_DHP = DHP_file[:,3] + 1j*DHP_file[:,4]

DHP_olg = in1_exc_DHP / in2_exc_DHP

DHP_C = np.loadtxt(dir+'DHARD_P_coherence_20230810.txt')
in1_exc_C_DHP = DHP_C[:,1]
in2_exc_C_DHP = DHP_C[:,2]

power = np.round(out["Px"] / 1e3)

N_avg = 20
# dhard p errorbars
dhard_p_mag_std = np.sqrt(csd_variance_mag(in1_exc_C_DHP, N_avg) + csd_variance_mag(in2_exc_C_DHP, N_avg))
dhard_p_phase_std = np.sqrt(csd_variance_phase(in1_exc_C_DHP, N_avg) + csd_variance_phase(in2_exc_C_DHP, N_avg))

fig=plt.figure()
ax=fig.add_subplot(211)
ax.loglog(freq_DHP, np.abs(DHP_olg), '.', label='Unbiased OLG taken at 60W PSL', color='xkcd:cornflower')
ax.fill_between(freq_DHP, np.abs(DHP_olg)*(1 + dhard_p_mag_std), np.abs(DHP_olg)*(1 - dhard_p_mag_std), color='xkcd:cornflower', alpha=0.3, label='$\pm 1$ $\sigma$' )
ax.loglog(
            F_Hz,
            np.abs(sol[names[0]][dofs_in[0], dofs_out[0]]),
            color=colors[0], label=f'Modeled with Finesse3, {power} kW'
        )
ax.set_title('DHARD P OLG')
ax.legend()
ax.set_xlim(0.1, 10)
ax.set_ylim(2e-3, 1e2)
ax.set_yticks([1e-2, 1, 1e2])
ax.set_ylabel('Mag')
ax.grid(True, which='both')

ax=fig.add_subplot(212)
ax.semilogx(freq_DHP, np.angle(DHP_olg, deg=True), '.', color='xkcd:cornflower')
ax.fill_between(freq_DHP, (180/np.pi)*(np.angle(DHP_olg) + dhard_p_phase_std), (180/np.pi)*np.angle(DHP_olg) - dhard_p_phase_std, color='xkcd:cornflower', alpha=0.3)
ax.semilogx(
            F_Hz,
            np.angle(sol[names[0]][dofs_in[0], dofs_out[0]], deg=True),
            color=colors[0],
        )
ax.set_ylabel('Phase [deg]')
ax.set_xlabel('Freq [Hz]')
ax.set_ylim(-200, 200)
ax.set_xlim(0.1, 10)
ax.grid(True, which='both')

# %%

N_avg = 25

CHP_file = np.loadtxt(dir+'CHARD_P_tf_20230810.txt')
freq_CHP = CHP_file[:,0]
in1_exc_CHP = CHP_file[:,1] + 1j*CHP_file[:,2]
in2_exc_CHP = CHP_file[:,3] + 1j*CHP_file[:,4]

CHP_olg = in1_exc_CHP / in2_exc_CHP

CHP_C = np.loadtxt(dir+'CHARD_P_coherence_20230810.txt')
in1_exc_C_CHP = CHP_C[:,1]
in2_exc_C_CHP = CHP_C[:,2]

# chard p errobars
chard_p_mag_std = np.sqrt(csd_variance_mag(in1_exc_C_CHP, N_avg) + csd_variance_mag(in2_exc_C_CHP, N_avg))
chard_p_phase_std = np.sqrt(csd_variance_phase(in1_exc_C_CHP, N_avg) + csd_variance_phase(in2_exc_C_CHP, N_avg))

fig=plt.figure()
ax=fig.add_subplot(211)
ax.loglog(freq_CHP, np.abs(CHP_olg), '.', label='Unbiased OLG taken at 60W PSL', color='xkcd:teal')
ax.fill_between(freq_CHP, np.abs(CHP_olg)*(1 + chard_p_mag_std), np.abs(CHP_olg)*(1 - chard_p_mag_std), color='xkcd:teal', alpha=0.3, label='$\pm 1$ $\sigma$' )
ax.loglog(
            F_Hz,
            np.abs(sol[names[1]][dofs_in[1], dofs_out[1]]),
            color=colors[1], label=f'Modeled with Finesse3, {power} kW'
        )
ax.set_title('CHARD P OLG')
ax.legend()
ax.set_xlim(0.1, 10)
ax.set_ylim(2e-3, 1e2)
ax.set_yticks([1e-2, 1, 1e2])
ax.set_ylabel('Mag')
ax.grid(True, which='both')

ax=fig.add_subplot(212)
ax.semilogx(freq_CHP, np.angle(CHP_olg, deg=True), '.', color='xkcd:teal')
ax.fill_between(freq_CHP, (180/np.pi)*(np.angle(CHP_olg) + chard_p_phase_std), (180/np.pi)*np.angle(CHP_olg) - chard_p_phase_std, color='xkcd:teal', alpha=0.3)
ax.semilogx(
            F_Hz,
            np.angle(sol[names[1]][dofs_in[1], dofs_out[1]], deg=True),
            color=colors[1],
        )
ax.set_ylabel('Phase [deg]')
ax.set_xlabel('Freq [Hz]')
ax.set_ylim(-200, 200)
ax.set_xlim(0.1, 10)
ax.grid(True, which='both')
# %%

N_avg = 25

CHY_file = np.loadtxt(dir+'CHARD_Y_tf.txt')
freq_CHY = CHY_file[:,0]
in1_exc_CHY = CHY_file[:,1] + 1j*CHY_file[:,2]
in2_exc_CHY = CHY_file[:,3] + 1j*CHY_file[:,4]

CHY_olg = in1_exc_CHY / in2_exc_CHY

CHY_C = np.loadtxt(dir+'CHARD_Y_coherence.txt')
in1_exc_C_CHY = CHY_C[:,1]
in2_exc_C_CHY = CHY_C[:,2]

# chard p errobars
chard_y_mag_std = np.sqrt(csd_variance_mag(in1_exc_C_CHY, N_avg) + csd_variance_mag(in2_exc_C_CHY, N_avg))
chard_y_phase_std = np.sqrt(csd_variance_phase(in1_exc_C_CHY, N_avg) + csd_variance_phase(in2_exc_C_CHY, N_avg))

fig=plt.figure()
ax=fig.add_subplot(211)
ax.loglog(freq_CHY, np.abs(CHY_olg), '.', label='Unbiased OLG taken at 60W PSL', color='xkcd:salmon')
ax.fill_between(freq_CHY, np.abs(CHY_olg)*(1 + chard_y_mag_std), np.abs(CHY_olg)*(1 - chard_y_mag_std), color='xkcd:salmon', alpha=0.3, label='$\pm 1$ $\sigma$' )
ax.loglog(
            F_Hz,
            np.abs(sol[names[3]][dofs_in[3], dofs_out[3]]),
            color=colors[3], label=f'Modeled with Finesse3, {power} kW'
        )
ax.set_title('CHARD Y OLG')
ax.legend()
ax.set_xlim(0.1, 10)
ax.set_ylim(2e-3, 1e2)
ax.set_yticks([1e-2, 1, 1e2])
ax.set_ylabel('Mag')
ax.grid(True, which='both')

ax=fig.add_subplot(212)
ax.semilogx(freq_CHY, np.angle(CHY_olg, deg=True), '.', color='xkcd:salmon')
ax.fill_between(freq_CHY, (180/np.pi)*(np.angle(CHY_olg) + chard_y_phase_std), (180/np.pi)*np.angle(CHY_olg) - chard_y_phase_std, color='xkcd:salmon', alpha=0.3)
ax.semilogx(
            F_Hz,
            np.angle(sol[names[3]][dofs_in[3], dofs_out[3]], deg=True),
            color=colors[3],
        )
ax.set_ylabel('Phase [deg]')
ax.set_xlabel('Freq [Hz]')
ax.set_ylim(-200, 200)
ax.set_xlim(0.1, 10)
ax.grid(True, which='both')
# %%
N_avg = 25

DHY_file = np.loadtxt(dir+'DHARD_Y_tf.txt')
freq_DHY = DHY_file[:,0]
in1_exc_DHY = DHY_file[:,1] + 1j*DHY_file[:,2]
in2_exc_DHY = DHY_file[:,3] + 1j*DHY_file[:,4]

DHY_olg = in1_exc_DHY / in2_exc_DHY

DHY_C = np.loadtxt(dir+'DHARD_Y_coherence.txt')
in1_exc_C_DHY = DHY_C[:,1]
in2_exc_C_DHY = DHY_C[:,2]

# DHard p errobars
dhard_y_mag_std = np.sqrt(csd_variance_mag(in1_exc_C_DHY, N_avg) + csd_variance_mag(in2_exc_C_DHY, N_avg))
dhard_y_phase_std = np.sqrt(csd_variance_phase(in1_exc_C_DHY, N_avg) + csd_variance_phase(in2_exc_C_DHY, N_avg))

fig=plt.figure()
ax=fig.add_subplot(211)
ax.loglog(freq_DHY, np.abs(DHY_olg), '.', label='Unbiased OLG taken at 60W PSL', color='xkcd:hot pink')
ax.fill_between(freq_DHY, np.abs(DHY_olg)*(1 + dhard_y_mag_std), np.abs(DHY_olg)*(1 - dhard_y_mag_std), color='xkcd:hot pink', alpha=0.3, label='$\pm 1$ $\sigma$' )
ax.loglog(
            F_Hz,
            np.abs(sol[names[2]][dofs_in[2], dofs_out[2]]),
            color=colors[0], label=f'Modeled with Finesse3, {power} kW'
        )
ax.set_title('DHARD Y OLG')
ax.legend()
ax.set_xlim(0.1, 10)
ax.set_ylim(2e-3, 1e2)
ax.set_yticks([1e-2, 1, 1e2])
ax.set_ylabel('Mag')
ax.grid(True, which='both')

ax=fig.add_subplot(212)
ax.semilogx(freq_DHY, np.angle(DHY_olg, deg=True), '.', color='xkcd:hot pink')
ax.fill_between(freq_DHY, (180/np.pi)*(np.angle(DHY_olg) + dhard_y_phase_std), (180/np.pi)*np.angle(DHY_olg) - dhard_y_phase_std, color='xkcd:hot pink', alpha=0.3)
ax.semilogx(
            F_Hz,
            np.angle(sol[names[2]][dofs_in[2], dofs_out[2]], deg=True),
            color=colors[0],
        )
ax.set_ylabel('Phase [deg]')
ax.set_xlabel('Freq [Hz]')
ax.set_ylim(-200, 200)
ax.set_xlim(0.1, 10)
ax.grid(True, which='both')
# %%
