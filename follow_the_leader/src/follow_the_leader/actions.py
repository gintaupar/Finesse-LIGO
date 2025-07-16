import traceback
from pathlib import Path
import shutil
from copy import deepcopy
import numpy as np
#from joblib import Memory
from finesse.exceptions import LostLock
from finesse.analysis.actions import (
    Maximize,
    Minimize,
    PseudoLockCavity,
    Series,
    Change,
    Noxaxis,
    OptimiseRFReadoutPhaseDC,
    SetLockGains,
    RunLocks,
    Execute,
)
from follow_the_leader.sim_data import (
    SimData
)
from follow_the_leader.io import (
    isYes
)
import pickle

#memory = Memory('ftl_sim.cache', verbose=1)

def correct_arm_loss_for_nonzero_maxtem(model, verbose=True):
    def myprint(*args, **kwargs):
        if verbose:
            return print(*args,**kwargs)
    
    eigx = model.run("eigenmodes(cavXARM, 0)")
    eigy = model.run("eigenmodes(cavYARM, 0)")

    loss_x = model.X_arm_loss + eigx.loss(True)[1][0]
    loss_y = model.Y_arm_loss + eigy.loss(True)[1][0]
    myprint("X arm loss: ", loss_x / 1e-6, "ppm")
    myprint("Y arm loss: ", loss_y / 1e-6, "ppm")
    # Apply corrections to get back to original losses
    myprint("Old X arm plane-wave loss: ", model.X_arm_loss / 1e-6, "ppm")
    myprint("Old Y arm plane-wave loss: ", model.Y_arm_loss / 1e-6, "ppm")
    model.X_arm_loss -= eigx.loss(True)[1][0]
    model.Y_arm_loss -= eigy.loss(True)[1][0]
    myprint("New X arm plane-wave loss: ", model.X_arm_loss / 1e-6, "ppm")
    myprint("New Y arm plane-wave loss: ", model.Y_arm_loss / 1e-6, "ppm")


def add_fields_ligo(model):
    model.parse(
    """
    mathd Parm (Px+Py)/2

    fd E_refl_c0  IFI.p4.o f=0
    fd E_refl_u9  IFI.p4.o f=+f1
    fd E_refl_l9  IFI.p4.o f=-f1
    fd E_refl_u45 IFI.p4.o f=+f2
    fd E_refl_l45 IFI.p4.o f=-f2
                        
    fd E_prc_c0  PRM.p1.o f=0
    fd E_prc_u9  PRM.p1.o f=+f1
    fd E_prc_l9  PRM.p1.o f=-f1
    fd E_prc_u45 PRM.p1.o f=+f2
    fd E_prc_l45 PRM.p1.o f=-f2
            
    fd E_src_c0  SRM.p1.o f=0
    fd E_src_u9  SRM.p1.o f=+f1
    fd E_src_l9  SRM.p1.o f=-f1
    fd E_src_u45 SRM.p1.o f=+f2
    fd E_src_l45 SRM.p1.o f=-f2
            
    fd E_x_c0  ETMX.p1.i f=0
    fd E_x_u9  ETMX.p1.i f=+f1
    fd E_x_l9  ETMX.p1.i f=-f1
    fd E_x_u45 ETMX.p1.i f=+f2
    fd E_x_l45 ETMX.p1.i f=-f2
            
    fd E_y_c0  ETMY.p1.i f=0
    fd E_y_u9  ETMY.p1.i f=+f1
    fd E_y_l9  ETMY.p1.i f=-f1
    fd E_y_u45 ETMY.p1.i f=+f2
    fd E_y_l45 ETMY.p1.i f=-f2
            
    fd E_inx_c0  ITMXlens.p1.i f=0
    fd E_inx_u9  ITMXlens.p1.i f=+f1
    fd E_inx_l9  ITMXlens.p1.i f=-f1
    fd E_inx_u45 ITMXlens.p1.i f=+f2
    fd E_inx_l45 ITMXlens.p1.i f=-f2
        
    fd E_iny_c0  ITMYlens.p1.i f=0
    fd E_iny_u9  ITMYlens.p1.i f=+f1
    fd E_iny_l9  ITMYlens.p1.i f=-f1
    fd E_iny_u45 ITMYlens.p1.i f=+f2
    fd E_iny_l45 ITMYlens.p1.i f=-f2

    fd E_c0_as OM1.p1.i f=0
"""
)


def MyLockLIGO(
    LSC_demod_opt=(
        "CARM",
        "REFL9_I",
        "PRCL",
        "POP9_I",
        "SRCL",
        "POP45_I",
        "DARM",
        "AS45_Q",
    ),
    run_locks=True,
    exception_on_lock_fail=True,
    exception_on_check_fail=True,
    gain_scale=0.5,
    lock_steps=2000,
    pseudo_lock_arms=False,
    break_point = 1000
):
    """Initial locking action, tries to somewhat replicate the locking prodcedure for
    LSC.

    If it can't find a good operating point then the RunLocks step at the end will fail.
    """

    def ALERT(x):
        if exception_on_check_fail:
            raise finesse_ligo.exceptions.InitialLockCheckException(x)
        else:
            warn(x)

    def check_arm_gains(state, name):
        X_GAIN = state.previous_solution["AGX"]
        Y_GAIN = state.previous_solution["AGY"]
        X_GAIN_APRX = 4 / state.model.ITMX.T
        Y_GAIN_APRX = 4 / state.model.ITMY.T

        if not (0.8 <= X_GAIN / X_GAIN_APRX <= 1.2):
            ALERT(
                f"X arm gain is significantly different to what was expected = {X_GAIN:.1f} vs {X_GAIN_APRX:.1f}"
            )
        if not (0.8 <= Y_GAIN / Y_GAIN_APRX <= 1.2):
            ALERT(
                f"Y arm gain is significantly different to what was expected = {Y_GAIN:.1f} vs {Y_GAIN_APRX:.1f}"
            )

    if pseudo_lock_arms:
        arm_locks = (
            PseudoLockCavity(
                "cavXARM", lowest_loss=False, feedback="XARM.DC", name="lock XARM"
            ),
            PseudoLockCavity(
                "cavYARM", lowest_loss=False, feedback="YARM.DC", name="lock YARM"
            ),
        )
    else:
        arm_locks = (
            Maximize("a_carrier_00_x", "XARM.DC", name="lock XARM"),
            Maximize("a_carrier_00_y", "YARM.DC", name="lock YARM"),
        )

    actions = []

    actions.append(    
        Change(
            {
                "ETMX.misaligned": False,
                "ITMX.misaligned": True,
                "ETMY.misaligned": False,
                "ITMY.misaligned": True,
                "SRM.misaligned": True,
                "PRM.misaligned": True,
                "SRM.misaligned": True,
                "CARM.DC": 0,
                "DARM.DC": 0,
                "SRCL.DC": 0,
                "PRCL.DC": 0,
                "MICH.DC": 0,
                "MICH2.DC": 0,
                "CARM_lock.enabled": False,
                "DARM_rf_lock.enabled": False,
                "SRCL_lock.enabled": False,
                "PRCL_lock.enabled": False,
                "MICH_lock.enabled": False,
            },
            name="misalign all",
        )
    )
    if break_point > 0:
        actions.extend([
            Change(
                {
                    "ETMX.misaligned": False,
                    "ITMX.misaligned": False,
                    "ETMY.misaligned": False,
                    "ITMY.misaligned": False,
                },
                name="misalign PRM,SRM",
            ),
            # Lock each arm cavity to the lowest loss mode
            *arm_locks,
            # Put mich on dark fringe
            Minimize("Pas_carrier", "MICH2.DC", name="find dark AS"),
            Noxaxis(name="after ARMs+AS"),
            Execute(check_arm_gains)
        ])

    if break_point > 1:
        actions.extend([     
            # Realign the PRM
            Change({"PRM.misaligned": False}, name="align PRM"),
            # get the PRC in roughly the right place whilst keeping arms on resonance
            Maximize("PRG", "PRCL.DC", name="maximise PRG"),
            # get the PRC in roughly the right place whilst keeping arms on resonance
            Maximize("cost_prcl", ["PRCL.DC", "CARM.DC"], name="maxmize Parm*PRG9"),
            Noxaxis(name="after PRC"),
            Execute(check_arm_gains),
            Change({"CARM.DC": 90}, relative=True),  # add 90 to get offset CARM for corner station locking
        ])

    if break_point > 2:
        actions.extend([ 
            Change({"CARM.DC": -90}, relative=True),  # Remove for locking procedure
            # Realign SRM
            Change({"SRM.misaligned": False}, name="align SRM"),
            # Try and find signal recycling condition
            Minimize("cost_srcl", "SRCL.DC", name="maximize SRC power"),
            Change({"SRCL.DC": 90}, relative=True),  # add 90 to get to RSE
            Minimize("cost_srcl", "SRCL.DC", name="minimize PRG45"),
            Noxaxis(name="after SRC"),
            Execute(check_arm_gains),
            Change({"CARM.DC": 90}, relative=True),  # add 90 to get offset CARM for corner station locking
        ])
    
    if break_point > 3:
        actions.extend([
            Change({"CARM.DC": -90}, relative=True),  # Remove for locking procedure
            Change(
                {
                    "CARM_lock.enabled": True,
                    "DARM_rf_lock.enabled": True,
                    "SRCL_lock.enabled": True,
                    "PRCL_lock.enabled": True,
                    "MICH_lock.enabled": True,
                },
                name="enable locks",
            ),
            OptimiseRFReadoutPhaseDC(*LSC_demod_opt),
            SetLockGains(d_dof_gain=1e-10, gain_scale=gain_scale),
        ])

    if run_locks:
        actions.extend([
            RunLocks(max_iterations=lock_steps, exception_on_fail=exception_on_lock_fail, name="final run locks"),
            Noxaxis(name="after locks"),
        ])
        if break_point == 1 or break_point > 3:
            # CARM is not offset
            actions.extend([
                Execute(check_arm_gains)
            ])
    
    action = Series(*tuple(actions))

    return action

def guard_state_to_lock_state(guard_state):
    state = guard_state
    if state < 100:
        # IFO has not yet found IR resonance in arms
        # We will leave finesse in an initial state
        # CARM=DARM=MICH=0
        return 0
    
    if state < 120:
        # IFO is looking for IR resonance in the arms
        # we will put finesse in arms locked
        # CARM & DARM controlled but corner misaligned
        return 1
    
    if state < 430:
        # Corner station is being locked
        # CARM is offset
        # We will put finesse with CARM offset
        # and corner station locked
        return 2
    
    if state < 520:
        # 520 is power up 2W
        return 2

    if state < 560:
        # Arms get brought on resonance at 560
        return 3

    return 4

# This didn't work due to the complex classes involved
#@memory.cache
def time_dependant_sim(TS, model, factory, Pin, G, comparison_time, max_time, lock,
                       cache_states=True, simDataKwargs={},
                       overwite_old_data_ok=False):

    llo = model
    time = [0]
    TS.values.out = llo.run()
    _lock_out = llo.run(lock)["final run locks"]
    last_lock_state = -1

    if cache_states:
        models = [llo.deepcopy()]
        locks = [_lock_out]
        values_used = [deepcopy(TS.values)]

    do_pickle = False # This is disabled, Pickling is too difficult
    if do_pickle:
        pickDir = Path(str(comparison_time))
        if pickDir.is_dir() and not overwite_old_data_ok:
            if not isYes('Previous simulation data exists. Overwrite?'):
                raise ValueError('Cannot overwrite previous data!')
        shutil.rmtree(pickDir, ignore_errors=True)
        pickDir.mkdir()

        def pick_fname():
            i = 0
            while True:
                yield pickDir / Path(f"{i:06}"+'.pkl')
                i += 1


        pfname = pick_fname()

        with open(next(pfname), "wb") as file:
            pickle.dump(
                {   "model": llo,
                    "values_used": TS.values
                },
                file,
            )

    output_matrix = simDataKwargs.pop("output_matrix", False)
    enable_CSV_output = simDataKwargs.pop("enable", True)
    if enable_CSV_output:
        out_data = SimData(comparison_time, **simDataKwargs)
        out_data.write_header(llo, TS.values.out, factory, 
                              output_matrix=output_matrix, 
                              exist_ok=overwite_old_data_ok)
        out_data.write_row(time[0], llo, TS.values.out)



    # Wrap in try except so if it dies, we still get data output
    try:
        TS.reset()

        while TS.t <= max_time:

            #################################
            # Choose step size
            #################################
            if TS.t < 31*60: #  For first 31 mins of simulation, use low step
                TS.dt = 20
            elif np.any(Pin(np.linspace(TS.t-5*60, TS.t, 5)) < 60): # If power has been less than 60W in in the last 5 minutes
                TS.dt = 20
            elif np.any(Pin(np.linspace(TS.t-30*60, TS.t, 5)) < 60): # If power has been less than 60W in in the last 30 minutes
                TS.dt = 100
            else:
                TS.dt = 200

            #################################
            # Step fowards and update maps
            #################################
            TS.step()
            TS.update_maps()
            
            # Load power data
            llo.L0.P = Pin(TS.t)
            lock_state = guard_state_to_lock_state(G(TS.t))
            
            #################################
            # Find operating point
            #################################
            _lock_out = None
            if lock_state != last_lock_state:
                print(f"switching to from lock state {last_lock_state} to {lock_state}")
                last_lock_state = lock_state
                
                # Update Finesse model to state in IFO
                lock = MyLockLIGO(
                    exception_on_lock_fail=True,
                    lock_steps=100,
                    gain_scale=0.4,
                    pseudo_lock_arms=False,
                    run_locks=True,
                    break_point = lock_state
                )
                try:
                    _lock_out = llo.run(lock)
                except IndexError:
                    # Sometimes we get an index error on state change
                    # it goes away if we just try again
                    _lock_out = llo.run(lock)
                    
                if cache_states:
                    locks.append(_lock_out["final run locks"])

                # update gains
                last_lock_power = llo.L0.P
                last_lock_gains = {}
                for lsc_dof in ["DARM_rf", "CARM", "PRCL", "SRCL", "MICH"]:
                    last_lock_gains[lsc_dof] = llo.get(f"{lsc_dof}_lock.gain")

            else:
                # We are in DFPMI mode
                # update gains for the input power
                for lsc_dof in ['DARM_rf', 'CARM', 'PRCL', 'SRCL', 'MICH']:
                    obj = getattr(llo, f'{lsc_dof}_lock')
                    obj.gain = last_lock_gains[lsc_dof] * last_lock_power / llo.L0.P

                try:
                    _lock_out = llo.run(RunLocks(exception_on_fail=True, max_iterations=1000))

                except LostLock:
                    print('Lock failed. Will retry locking from scratch on the next cycle')
                    last_lock_state = None

                finally:
                    if cache_states:
                        locks.append(_lock_out)
            

            #################################
            # Run simulation and store data
            #################################
            time.append(TS.t)
            TS.values.out = llo.run(Noxaxis())

            if cache_states:
                models.append(llo.deepcopy())
                values_used.append(deepcopy(TS.values))

            if enable_CSV_output:
                out_data.write_row(time[-1], llo, TS.values.out)

            if do_pickle:
                with open(next(pfname), "wb") as file:
                    pickle.dump(
                        {   "model": llo,
                            "values_used": TS.values
                        },
                        file,
                    )

            ###################
            # Output to STDERR
            ###################
            PRG = TS.values.out["PRG"]
            print(f'T={TS.t:5.0f}, Pin={llo.L0.P.value:3.3f} W, '
                f'Parm={TS.values.out["Parm"]/1e3:4.2f} kW, PRG={PRG:3.1f}, '
                f'fx={float(llo.ITMXlens.f.value)/1e3:6.0f} km, fy={float(llo.ITMYlens.f.value)/1e3:6.0f} km')
    
    except Exception as error:
        traceback.print_exc()
    
    if cache_states:
        return {'models': models, 
                'values_used': values_used, 'locks': locks}
    else:
        return {'TS': TS, 'model': llo}
