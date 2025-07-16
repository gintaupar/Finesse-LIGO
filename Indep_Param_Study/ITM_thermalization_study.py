#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, time, pickle, finesse, argparse
import finesse.ligo
import numpy as np
import finesse.analysis.actions as fac
from tqdm import tqdm
from finesse.knm import Map
from finesse.ligo.factory import ALIGOFactory
from finesse.ligo.actions import InitialLockLIGO
from finesse.ligo.maps import (get_test_mass_surface_profile_interpolated,
                               aligo_O4_TM_aperture, aligo_O4_ESD_inner_aperture,
                               aligo_O4_PR3_SR3_baffle)

parentrepo = "/home/shreejit.jadhav/WORK/RC/ligo-commissioning-modeling"
sys.path.append(f"{parentrepo}/analysis/O4/GA_State_Optimization/src")

from funcs import (ALIGOBeamSplitterFactory,
                  llo_O4_HR_baffle,
                  llo_O4_BS_AR_baffle,
                  llo_O4_BS_HR_aperture,
                  llo_O4_BS_AR_aperture,
                  set_cold_state_beam,
                  )


def set_parameters(llo, p_itmx_Rc, p_itmy_Rc, p_itmx_f, p_itmy_f):
    llo.ITMXlens.f.value = 1/(1/base.ITMXlens.f.eval() + p_itmx_f)
    llo.ITMYlens.f.value = 1/(1/base.ITMYlens.f.eval() + p_itmy_f)
    llo.ITMX.Rc = 2/(2/base.ITMX.Rcx.eval() + p_itmx_Rc)
    llo.ITMY.Rc = 2/(2/base.ITMY.Rcx.eval() + p_itmy_Rc)
    llo.ETMX.Rc = 2/(2/base.ETMX.Rcx.eval() + p_itmx_Rc)
    llo.ETMY.Rc = 2/(2/base.ETMY.Rcx.eval() + p_itmy_Rc)


def process_lens_and_surface_deformations(llo, base, data, state=None, stepfac=1, skipsteptrig=8):
    """
    Adjusts the lens and surface deformations and collects data.

    Parameters:
    - llo: The LLO model object.
    - base: The base model object for reference.
    - data: Dictionary to store the results.
    - state: contains dicts of current params and target params.
    - stepfac: factor deciding num of steps to reach the given params from prev params.
               Intermediate steps of 'stepsize = (current_params - prev_params)/stepfac'
               will be taken.

    Returns:
    - Updated data dictionary with new measurements.
    - Parameters for current and target states
    """

    i = 0
    for p_itmx_Rc, p_itmy_Rc, p_itmx_f, p_itmy_f in zip(np.linspace(state["current"]["p_itmx_Rc"], state["target"]["p_itmx_Rc"], stepfac+1)[1:],
                                                        np.linspace(state["current"]["p_itmy_Rc"], state["target"]["p_itmy_Rc"], stepfac+1)[1:],
                                                        np.linspace(state["current"]["p_itmx_f"], state["target"]["p_itmx_f"], stepfac+1)[1:],
                                                        np.linspace(state["current"]["p_itmy_f"], state["target"]["p_itmy_f"], stepfac+1)[1:],
                                                        ):

        print(i, p_itmx_Rc, p_itmy_Rc, p_itmx_f, p_itmy_f)
        i += 1

        if stepfac >= skipsteptrig:
            print("Skipping forward..")
            continue

        # Set initial parameters
        set_parameters(llo, p_itmx_Rc, p_itmy_Rc, p_itmx_f, p_itmy_f)

        llo.beam_trace()
        # llo.run(fac.SetLockGains(gain_scale=0.4))

        try:
            sols = llo.run("series(run_locks(exception_on_fail=True, max_iterations=5000), noxaxis())")

            # update current state
            state["current"]["p_itmx_Rc"] = p_itmx_Rc
            state["current"]["p_itmy_Rc"] = p_itmy_Rc
            state["current"]["p_itmx_f"] = p_itmx_f
            state["current"]["p_itmy_f"] = p_itmy_f
            stepfac -= 1

            # Inputs
            data["ITMXlens.f"].append(llo.ITMXlens.f.eval())
            data["ITMYlens.f"].append(llo.ITMYlens.f.eval())
            data["ITMX.Rcx"].append(llo.ITMX.Rcx.eval())
            data["ITMY.Rcx"].append(llo.ITMY.Rcx.eval())
            data["ETMX.Rcx"].append(llo.ETMX.Rcx.eval())
            data["ETMY.Rcx"].append(llo.ETMY.Rcx.eval())

            # Outputs
            data["gouy"].append(np.mean([llo.cavPRX.gouy/2, llo.cavPRY.gouy/2]))
            data["PRG"].append(sols["noxaxis"]["PRG"])
            data["PRG9"].append(sols["noxaxis"]["PRG9"])
            data["PRG45"].append(sols["noxaxis"]["PRG45"])
            data["PreflPRM"].append(sols["noxaxis"]["PreflPRM"])
            data["Px"].append(sols["noxaxis"]["Px"])

            print("### Gouy: ", data["gouy"][-1], "PRG: ", data["PRG"][-1], "PRG9: ", data["PRG9"][-1], "PRG45: ", data["PRG45"][-1], "PreflPRM: ", data["PreflPRM"][-1])

        except Exception as e:
            print("Initial run failed, attempting progressive steps:", e)
            stepfac *= 2

            print(f"Updated #steps to target state: {stepfac}")
            process_lens_and_surface_deformations(llo, base, data, state=state, stepfac=stepfac)

    return data, state


parser = argparse.ArgumentParser(description='ITM Thermalization Study')
parser.add_argument('--index', type=int, required=True, help='Run index')
parser.add_argument('--p_itmx_Rc', type=float, required=True, help='Curvature power addition for ITMX Rc (ppm)')
parser.add_argument('--p_itmy_Rc', type=float, required=True, help='Curvature power addition for ITMY Rc (ppm)')
parser.add_argument('--p_min_itmx_f', type=float, required=True, help='Min ITMX lens power variation (ppm)')
parser.add_argument('--p_max_itmx_f', type=float, required=True, help='Max ITMX lens power variation (ppm)')
parser.add_argument('--p_min_itmy_f', type=float, required=True, help='Min ITMY lens power variation (ppm)')
parser.add_argument('--p_max_itmy_f', type=float, required=True, help='Max ITMY lens power variation (ppm)')
parser.add_argument('--N', type=int, required=True, help='Number of steps of ITM lens variation')
parser.add_argument('--yamlpath', type=str, required=True, help='Path to yaml file')
parser.add_argument('--suffix', type=str, required=True, help='Path to yaml file')
args = parser.parse_args()

print(args)

suffix = args.suffix
repopath = f"{parentrepo}/analysis/O4/Indep_Param_Study"

maxtems = 8
datapath = f'{repopath}/data/run_{suffix}'
update_yaml = parentrepo + args.yamlpath
w0 = 1.015e-3
z = 6.0
is_astig = True
new_bs = True
add_BS_baffle = True
addl_parse = None
Nsteps = args.N

# lens optical powers in D
P_ITMX_LOW, P_ITMX_HIGH = args.p_min_itmx_f * 1e-6, args.p_max_itmx_f * 1e-6
P_ITMY_LOW, P_ITMY_HIGH = args.p_min_itmy_f * 1e-6, args.p_max_itmy_f * 1e-6
p_itmx_Rc, p_itmy_Rc = args.p_itmx_Rc * 1e-6, args.p_itmy_Rc * 1e-6

p_itmx_fs = np.linspace(P_ITMX_LOW, P_ITMX_HIGH, Nsteps)
p_itmy_fs = np.linspace(P_ITMY_LOW, P_ITMY_HIGH, Nsteps)

if not os.path.exists(datapath):
    os.makedirs(datapath)

if new_bs:
    factory = ALIGOBeamSplitterFactory(f"{parentrepo}/LLO/yaml/llo_O4.yaml")
else:
    factory = ALIGOFactory(f"{parentrepo}/LLO/yaml/llo_O4.yaml")
factory.update_parameters(f"{parentrepo}/LLO/yaml/llo_addRH.yaml")

if update_yaml:
    factory.update_parameters(update_yaml)

factory.params.INPUT.LASER.power = 2

# Make the model
factory.reset() # always reset to default
factory.options.LSC.add_locks = True
factory.options.LSC.add_output_detectors = True
factory.options.ASC.add = False
factory.options.INPUT.add_IMC_and_IM1 = False
factory.options.thermal.add = True

llo = factory.make()

if addl_parse is not None:
    llo.parse(addl_parse)

if new_bs:
    if add_BS_baffle:
        xBS, BS_HRPR_aperture = llo_O4_HR_baffle()
        _, BS_HRY_aperture = llo_O4_HR_baffle()
        xBSARX, BS_ARX_aperture = llo_O4_BS_AR_baffle(offset_direction=-1)
        xBSARAS, BS_ARAS_aperture = llo_O4_BS_AR_baffle(offset_direction=1)

        llo.BSHRPR.surface_map = Map(xBS, xBS, amplitude = BS_HRPR_aperture)
        llo.BSHRY.surface_map = Map(xBS, xBS, amplitude = BS_HRY_aperture)
        llo.BSARX.surface_map = Map(xBSARX, xBSARX, amplitude = BS_ARX_aperture)
        llo.BSARAS.surface_map = Map(xBSARAS, xBSARAS, amplitude = BS_ARAS_aperture)
    else:
        xBS, BS_HR_aperture = llo_O4_BS_HR_aperture()
        xBSARX, BS_ARX_aperture = llo_O4_BS_AR_aperture(offset_direction=-1)
        xBSARAS, BS_ARAS_aperture = llo_O4_BS_AR_aperture(offset_direction=1)
        llo.BS.surface_map = Map(xBS, xBS, amplitude = BS_HR_aperture)
        llo.BSARX.surface_map = Map(xBSARX, xBSARX, amplitude = BS_ARX_aperture)
        llo.BSARAS.surface_map = Map(xBSARAS, xBSARAS, amplitude = BS_ARAS_aperture)

xPR3_SR3, PR3_SR3_aperture = aligo_O4_PR3_SR3_baffle()
llo.PR3.surface_map = Map(xPR3_SR3, xPR3_SR3, amplitude = PR3_SR3_aperture)
llo.SR3.surface_map = Map(xPR3_SR3, xPR3_SR3, amplitude = PR3_SR3_aperture)

llo = set_cold_state_beam(llo, w0, z, update_factory=factory, waist_params={}, losses={}, is_astig=is_astig)

base = llo.deepcopy()

# set the 1st set of lenses and surface deformations
set_parameters(llo, p_itmx_Rc, p_itmy_Rc, p_itmx_fs[0], p_itmy_fs[0])

# Run cold state model to set starting point
R = 0.17
N = 201 
x, TM_aperture = aligo_O4_TM_aperture(R, N)
x, lens_aperture = aligo_O4_ESD_inner_aperture(R, N)
y = x

# Get surfaces
for TM in [llo.ITMX, llo.ITMY, llo.ETMX, llo.ETMY]:
    if "X" in TM.name:
        P = factory.params.X
    else:
        P = factory.params.Y
    TM._unfreeze()
    TM.static = get_test_mass_surface_profile_interpolated(P[TM.name[:-1]].ID, make_axisymmetric=True)(x, y)
    TM.aperture = TM_aperture
    TM._freeze()

    TM.surface_map = Map(x, y, amplitude = TM.aperture, opd=TM.static)

for TMlens in [llo.ITMXlens, llo.ITMYlens]:
    TMlens._unfreeze()
    TMlens.aperture = lens_aperture
    TMlens._freeze()
    TMlens.OPD_map = Map(x, y, amplitude = TMlens.aperture)

# compute the round trip losses with the maps in and make sure overall loss is reasonable
llo.modes("even", maxtem=maxtems)
eigx = llo.run("eigenmodes(cavXARM, 0)")
eigy = llo.run("eigenmodes(cavYARM, 0)")

# Apply corrections to get back to original losses
llo.X_arm_loss -= eigx.loss(True)[1][0]
llo.Y_arm_loss -= eigy.loss(True)[1][0]

lock = InitialLockLIGO(
    exception_on_lock_fail=False,
    exception_on_check_fail=False,
    lock_steps=50000,
    gain_scale=0.6,
    pseudo_lock_arms=False,
    run_locks=True,
)

sol = llo.run(fac.Series(lock, fac.Noxaxis(name="noxaxis")))
print(
    "Gouy", np.mean([llo.cavPRX.gouy/2, llo.cavPRY.gouy/2]),
    "Px", sol["noxaxis"]["Px"],
    "PRG", sol["noxaxis"]["PRG"],
    "PRG9", sol["noxaxis"]["PRG9"]
)

# save initial gains to be changed during power up
initial_gains = {}
for lsc_dof in ["DARM_rf", "CARM", "PRCL", "SRCL", "MICH"]:
    initial_gains[lsc_dof] = llo.get(f"{lsc_dof}_lock.gain")


llo.L0.P = 2
llo.beam_trace()

data = {"ITMXlens.f": [],
        "ITMYlens.f": [],
        "ITMX.Rcx": [],
        "ITMY.Rcx": [],
        "ETMX.Rcx": [],
        "ETMY.Rcx": [],
        "gouy": [],
        "PRG": [],
        "PRG9": [],
        "PRG45": [],
        "PreflPRM": [],
        "Px": [],
}

state = {
    "current": {
        "p_itmx_Rc": p_itmx_Rc,
        "p_itmy_Rc": p_itmy_Rc,
        "p_itmx_f": p_itmx_fs[0],
        "p_itmy_f": p_itmy_fs[0],
        },
    }

program_starts = time.time()

signj = 1
skip = 1
for i in tqdm(range(len(p_itmx_fs))):
    p_itmx_f = p_itmx_fs[i]
    for j in range(len(p_itmy_fs)):
        p_itmy_f = p_itmy_fs[::signj][j]

        if skip == 1:
            skip = 0
            continue

        state["target"] = {
            "p_itmx_Rc": p_itmx_Rc,
            "p_itmy_Rc": p_itmy_Rc,
            "p_itmx_f": p_itmx_f,
            "p_itmy_f": p_itmy_f,
            }

        data, state = process_lens_and_surface_deformations(
            llo, base, data, state, stepfac=1,
            )

    signj *= -1

end_times = time.time()
print(f"Elapsed time: {end_times - program_starts} s")

for k in data:
    data[k] = np.array(data[k])

# save data
filepath = datapath+f"/data_{args.index}_{suffix}.pkl"
print(f"saving data to {filepath}")
with open(filepath, "wb") as file:
    pickle.dump(
        data,
        file,
    )

print("Data saved at {filepath}")

