# %%
import finesse.ligo
import numpy as np
import matplotlib.pyplot as plt
from munch import Munch
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
factory.options.thermal.add = True
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

factory.ASC_output_matrix.CHARD_P = Munch(
    {"ITMX": -0.74, "ETMX": +1, "ITMY": -0.74, "ETMY": +1}
)
factory.ASC_output_matrix.DHARD_P = Munch(
    {"ITMX": -0.74, "ETMX": +1, "ITMY": +0.74, "ETMY": -1}
)
factory.ASC_output_matrix.CSOFT_P = Munch(
    {"ITMX": +1, "ETMX": +0.74, "ITMY": +1, "ETMY": +0.74}
)
factory.ASC_output_matrix.DSOFT_P = Munch(
    {"ITMX": +1, "ETMX": +0.74, "ITMY": -1, "ETMY": -0.74}
)
factory.ASC_output_matrix.CHARD_Y = Munch(
    {"ITMX": +0.72, "ETMX": 1, "ITMY": -0.72, "ETMY": -1}
)
factory.ASC_output_matrix.DHARD_Y = Munch(
    {"ITMX": +0.72, "ETMX": +1, "ITMY": +0.72, "ETMY": +1}
)
factory.ASC_output_matrix.CSOFT_Y = Munch(
    {"ITMX": +1, "ETMX": -0.72, "ITMY": -1, "ETMY": +0.72}
)
factory.ASC_output_matrix.DSOFT_Y = Munch(
    {"ITMX": +1, "ETMX": -0.72, "ITMY": +1, "ETMY": -0.72}
)

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
P_target = 1e3*np.round(np.linspace(10,430, 25))
model = factory.make()
model.modes("even", maxtem=4)
model.run(fac.Series(InitialLockLIGO(), DARM_RF_to_DC()))

DHARD_P = model.DHARD_P
DHARD_Y = model.DHARD_Y
CHARD_P = model.CHARD_P
CHARD_Y = model.CHARD_Y

F_Hz = np.geomspace(0.1, 10, 200)
# %%
freq_resps = []
Pins = np.linspace(2, 70, 10)
for i, Pin in enumerate(Pins):
    print("Pin", i)
    j = 0
    model.L0.P = Pin
    model.run("set_lock_gains()")
    out = model.run()
    Parm = (out['Px'] + out['Py'])/2

    P_arm_prev = np.inf
    while abs(P_arm_prev - Parm) > 0.02 * abs(Parm):
        print(j, "Px", out['Px'], "Py", out['Py'], "Diff", abs(P_arm_prev - Parm))
        # Rewrite, make it do a quasistatic power up starting from 2W -> some target
        # Each step reoptimise locks
        P_arm_prev = Parm
        model.P_XARM = out['Px']
        model.P_YARM = out['Py']
        model.run("run_locks()")
        out = model.run()
        Parm = (out['Px'] + out['Py'])/2
        j += 1

    print("Done", P_arm_prev - Parm)
    sol1 = model.run(
        FrequencyResponse(
            F_Hz,
            [DHARD_P.AC.i, CHARD_P.AC.i, DHARD_Y.AC.i, CHARD_Y.AC.i],
            [DHARD_P.AC.o, CHARD_P.AC.o, DHARD_Y.AC.o, CHARD_Y.AC.o],
        )
    )
    freq_resps.append(sol1)

# %%
