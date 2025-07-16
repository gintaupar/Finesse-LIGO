import sys
import time
import pickle
import finesse
import finesse.ligo
import numpy as np
import matplotlib.pyplot as plt
import finesse.components as fc
import finesse.analysis.actions as fac
from copy import deepcopy
from finesse.knm import Map
from finesse.materials import FusedSilica
from finesse.ligo.factory import aligo, ALIGOFactory
from finesse.ligo.actions import InitialLockLIGO
from finesse.ligo.maps import (get_test_mass_surface_profile_interpolated,
                               aligo_O4_TM_aperture, aligo_O4_ESD_inner_aperture,
                               aligo_O4_PR3_SR3_baffle)
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(str(finesse.ligo.git_path() / "LLO"))

from thermal_rom import (I_ITMX_HR,
                         I_ETMX_HR,
                         I_ITMY_HR,
                         I_ETMY_HR,
                         compute_new_opd_state,
                         compute_new_deformation_state,
                         update_maps,
                         make_ts_optics,
                         add_thermal_detectors,
                         SimpleNamespace,
                         )

class ALIGOBeamSplitterFactory(aligo.ALIGOFactory):
    def add_BS(self, model, params):
        """
        Adding a new realistic BS in which port 1 and 2 of BS HR
        are each connected to a mirror element on which separate
        maps can be applied
        """

        # BS-HR
        BS = model.add(
            fc.Beamsplitter(
                "BS",
                T = params.T,
                L = params.L,
                alpha = params.AOI
            )
        )

        self.options.BS_trans_arm = self.options.BS_trans_arm.upper()
        assert self.options.BS_trans_arm in ("X", "Y")
        trans_arm = self.options.BS_trans_arm
        if trans_arm == "X":
            refl_arm = "Y"
        else:
            refl_arm = "X"

        # Baffle mirror component
        BSHRPR = model.add(
            fc.Mirror(
                "BSHRPR",
                T = 1,
                L = 0,
                phi = BS.phi.ref,
                xbeta = BS.xbeta.ref,
                ybeta = BS.ybeta.ref,
            )
        )
        BSHRarm = model.add(
            fc.Mirror(
                f"BSHR{refl_arm}",
                T = 1,
                L = 0,
                phi = BS.phi.ref,
                xbeta = BS.xbeta.ref,
                ybeta = BS.ybeta.ref,
            )
        )

        # Adding AR surface
        alpha_AR,  L_BS_sub = self._BS_substrate_length(BS, params)

        # Xarm AR
        BSARarm = model.add(
            fc.Beamsplitter(
                    f"BSAR{trans_arm}",
                    R=0,
                    L=params.R_AR,
                    alpha=np.rad2deg(alpha_AR),
                    phi=BS.phi.ref,
                    xbeta=BS.xbeta.ref,
                    ybeta=BS.ybeta.ref,
                )
        )

        # AS AR:
        BSARSR = model.add(
                fc.Beamsplitter(
                    "BSARAS",
                    R=0,
                    L=params.R_AR,
                    alpha=np.rad2deg(alpha_AR),
                    phi=BS.phi.ref,
                    xbeta=BS.xbeta.ref,
                    ybeta=BS.ybeta.ref,
                )
            )

        # Connect all components:
        # PR aperature to BS:
        model.connect(
            BSHRPR.p2,
            BS.p1,
            L = 0,
        )
        model.connect(
            BS.p2,
            BSHRarm.p1,
            L = 0,
        )
        model.connect(
            BS.p3,
            BSARarm.p1,
            L = L_BS_sub,
            name = f"subBS_{trans_arm}",
            nr = FusedSilica.nr,
        )
        model.connect(
            BS.p4,
            BSARSR.p2,
            L = L_BS_sub,
            name = "subBS_SR",
            nr = FusedSilica.nr
        )

        # Connecting mechanical dofs
        # self._link_all_mechanical_dofs(BS, BSARarm)
        # self._link_all_mechanical_dofs(BS, BSARSR)
        # self._link_all_mechanical_dofs(BS, BSHRarm)
        # self._link_all_mechanical_dofs(BS, BSHRPR)
        model.connect(BS.mech.z, BSARarm.mech.z, gain=1)
        model.connect(BS.mech.pitch, BSARarm.mech.pitch, gain=1)
        model.connect(BS.mech.yaw, BSARarm.mech.yaw)

        model.connect(BS.mech.z, BSARSR.mech.z, gain=1)
        model.connect(BS.mech.pitch, BSARSR.mech.pitch, gain=1)
        model.connect(BS.mech.yaw, BSARSR.mech.yaw)

        model.connect(BS.mech.z, BSHRarm.mech.z, gain=1)
        model.connect(BS.mech.pitch, BSHRarm.mech.pitch, gain=1)
        model.connect(BS.mech.yaw, BSHRarm.mech.yaw)

        model.connect(BS.mech.z, BSHRPR.mech.z, gain=1)
        model.connect(BS.mech.pitch, BSHRPR.mech.pitch, gain=1)
        model.connect(BS.mech.yaw, BSHRPR.mech.yaw)

        if trans_arm == "X":
            self.BS_X  = BSARarm.p3
            self.BS_Y = BSHRarm.p2
            self.BS_HR_X =  BS.p3
            self.BS_HR_Y = BSHRarm.p2
        else:
            self.BS_X =  BSHRarm.p2
            self.BS_Y  = BSARarm.p3
            self.BS_HR_X  = BSHRarm.p2
            self.BS_HR_Y = BS.p3

        self.BS_PR = BSHRPR.p1
        self.BS_SR = BSARSR.p4

def print_params(newpop, i):
    for path, item in newpop.items():
        p = path.split('/')

        if p[0] in ('beam_waist', 'absorption', 'RH_eff'):
            print(path, item['vals'][i], f"dtype: {item['vals'][i].dtype}")
            continue
        elif p[0] == 'loss':
            if p[1] == 'PRCL':
                print('PRCL loss', item['vals'][i] * 1e-6, f"dtype: {item['vals'][i].dtype}")
            else:
                print('losses', p[1], item['vals'][i], f"dtype: {item['vals'][i].dtype}")
            continue

        if len(p) == 2:
            if newpop[path]['type'] == 'abs':
                print(p[0], p[1], item['vals'][i], f"dtype: {item['vals'][i].dtype}")
            elif newpop[path]['type'] == 'rel':
                print(p[0], p[1], 1 + item['vals'][i], f"dtype: {item['vals'][i].dtype}")
        elif len(p) == 3:
            if newpop[path]['type'] == 'abs':
                print(p[0], p[1], p[2], item['vals'][i], f"dtype: {item['vals'][i].dtype}")
            elif newpop[path]['type'] == 'rel':
                print(p[0], p[1], p[2], 1 + item['vals'][i], f"dtype: {item['vals'][i].dtype}")

def apply_param_variations(factory, param_variations, i):

    waist_params = {}
    losses = {}
    TM_absorptions = {'ITMX': 0.5e-6, 'ITMY': 0.5e-6, 'ETMX': 0.3e-6, 'ETMY': 0.3e-6}
    RH_efficiency = {'ITMX': 0.9, 'ITMY': 0.9, 'ETMX': 0.9, 'ETMY': 0.9}

    for path, item in param_variations.items():
        p = path.split('/')

        if p[0] == 'beam_waist':
            print(f"new Beam waist {p[1]}: {item['vals'][i]}")
            waist_params[p[1]] = item['vals'][i]
            continue

        elif p[0] == 'loss':
            if p[1] == 'PRCL':
                print(f"orig PRCL loss: {factory.params['PRC']['PR3']['L']}")
                factory.params['PRC']['PR3']['L'] = item['vals'][i] * 1e-6
                if factory.params['PRC']['PR3']['L'] < 0:
                    factory.params['PRC']['PR3']['L'] = 0
                print(f"new PRCL loss: {factory.params['PRC']['PR3']['L']}")
            else:
                print(f"new Loss {p[1]}: {item['vals'][i]}")
                losses[p[1]] = item['vals'][i]
                if losses[p[1]] < 0:
                    losses[p[1]] = 0
            continue

        elif p[0] == 'absorption':
            print(f"new Absorption {p[1]}: {item['vals'][i]}")
            TM_absorptions[p[1]] = item['vals'][i]
            if TM_absorptions[p[1]] < 0:
                TM_absorptions[p[1]] = 0
            continue

        elif p[0] == 'RH_eff':
            print(f"new RH efficiency {p[1]}: {item['vals'][i]}")
            RH_efficiency[p[1]] = item['vals'][i]
            if RH_efficiency[p[1]] < 0:
                # allowing negative values of RH_efficiency as well
                # to accommodate crazy additional lenses at ITM
                pass
            continue

        if len(p) == 2:
            print(f"old {p[0]} {p[1]}: {factory.params[p[0]][p[1]]}")
            if param_variations[path]['type'] == 'abs':
                factory.params[p[0]][p[1]] += item['vals'][i]
                print(f"new {p[0]} {p[1]}: {factory.params[p[0]][p[1]]}")
            elif param_variations[path]['type'] == 'rel':
                factory.params[p[0]][p[1]] *= 1 + item['vals'][i]
                print(f"new {p[0]} {p[1]}: {factory.params[p[0]][p[1]]}")
        elif len(p) == 3:
            print(f"old {p[0]} {p[1]} {p[2]}: {factory.params[p[0]][p[1]][p[2]]}")
            if param_variations[path]['type'] == 'abs':
                factory.params[p[0]][p[1]][p[2]] += item['vals'][i]
                print(f"new {p[0]} {p[1]} {p[2]}: {factory.params[p[0]][p[1]][p[2]]}")
            elif param_variations[path]['type'] == 'rel':
                factory.params[p[0]][p[1]][p[2]] *= 1 + item['vals'][i]
                print(f"new {p[0]} {p[1]} {p[2]}: {factory.params[p[0]][p[1]][p[2]]}")

    return factory, waist_params, losses, TM_absorptions, RH_efficiency

def undo_param_variations(factory, param_variations, i):
    # undo the variations
    for path, item in param_variations.items():
        p = path.split('/')

        if p[0] in ('beam_waist', 'loss', 'absorption', 'RH_eff'):
            continue

        if len(p) == 2:
            if param_variations[path]['type'] == 'abs':
                factory.params[p[0]][p[1]] -= item['vals'][i]
                print(f"(param var undone) {p[0]} {p[1]}: {factory.params[p[0]][p[1]]}")
            elif param_variations[path]['type'] == 'rel':
                factory.params[p[0]][p[1]] /= 1 + item['vals'][i]
                print(f"(param var undone) {p[0]} {p[1]}: {factory.params[p[0]][p[1]]}")
        elif len(p) == 3:
            if param_variations[path]['type'] == 'abs':
                factory.params[p[0]][p[1]][p[2]] -= item['vals'][i]
                print(f"(param var undone) {p[0]} {p[1]} {p[2]}: {factory.params[p[0]][p[1]][p[2]]}")
            elif param_variations[path]['type'] == 'rel':
                factory.params[p[0]][p[1]][p[2]] /= 1 + item['vals'][i]
                print(f"(param var undone) {p[0]} {p[1]} {p[2]}: {factory.params[p[0]][p[1]][p[2]]}")

    return factory

# record the results
def record_sol(times, runsols, include_sol_at, outs):

    if include_sol_at is not None:
        irec = []
        for t1 in include_sol_at:
            irec.append(np.argmin(np.abs(np.array(times) - t1)))
        sol_rec = [outs[i] for i in irec]

    for k in sol_rec[-1].outputs:
        if not isinstance(sol_rec[-1][k], np.ndarray):
            if k not in runsols:
                runsols[k] = np.array([sol[k] for sol in sol_rec]).reshape(1, -1)
            else:
                runsols[k] = np.append(runsols[k], np.array([sol[k] for sol in sol_rec]).reshape(1, -1), axis=0)

    return runsols

# record model parameters
def record_model_params(times, models, runsols, model_output_params, include_sol_at):

    if include_sol_at is not None:
        irec = []
        for t1 in include_sol_at:
            irec.append(np.argmin(np.abs(np.array(times) - t1)))
        models_rec = [models[i] for i in irec]

    # record model params
    for k in model_output_params:
        tmparray = []
        for model in models_rec:
            model.beam_trace()
            p2 = k.split('.')

            if k == 'gouy':
                gouyphase = np.mean([model.cavPRX.gouy/2, model.cavPRY.gouy/2])
                if np.isnan(gouyphase):
                    gouyphase = 0
                tmparray.append(gouyphase)
                # print(f"Recorded gouy phase {runsols['gouy'][-1]}")
            elif k == 'gouy_xarm':
                tmparray.append(np.mean(model.cavXARM.gouy)/2)
            elif k == 'gouy_yarm':
                tmparray.append(np.mean(model.cavYARM.gouy)/2)
            else:
                tmparray.append(model.elements[p2[0]].nodes[k].qx.w)

        if k not in runsols:
            runsols[k] = np.array(tmparray).reshape(1, -1)
        else:
            runsols[k] = np.append(runsols[k], np.array(tmparray).reshape(1, -1), axis=0)
    return runsols

def get_astig_params(w0, z, is_astig):
    if is_astig:
        wkey = 'w0'
        zkey = 'z'
        astigParams = {'w0x': w0, 'zx': z, 'w0y': w0, 'zy': z}
    else:
        wkey = 'w0x'
        zkey = 'zx'
        astigParams = {'w0': w0, 'z': z}

    return wkey, zkey, astigParams

def runligo(factory,
            RH_efficiency={'ITMX': 0.9, 'ITMY': 0.9, 'ETMX': 0.9, 'ETMY': 0.9},
            w0=None,
            z=None,
            waist_params={},
            losses={},
            is_astig=False,
            maxtems=8,
            model_output_params=None,
            runsols=None,
            ):

    llo = factory.make()

    # override RH_efficiency if present in factory params
    if 'RH_eff' in factory.params:
        for TMkey in factory.params.RH_eff:
            RH_efficiency[TMkey] = factory.params.RH_eff[TMkey]

    # ring heater powers [W]
    P_RH_ITMX = 2*factory.params.P_RH_ITMX * RH_efficiency['ITMX']
    P_RH_ITMY = 2*factory.params.P_RH_ITMY * RH_efficiency['ITMY']
    P_RH_ETMX = 2*factory.params.P_RH_ETMX * RH_efficiency['ETMX']
    P_RH_ETMY = 2*factory.params.P_RH_ETMY * RH_efficiency['ETMY']

    set_ringheaters(llo, factory, "X", P_RH_ITMX, P_RH_ETMX)
    set_ringheaters(llo, factory, "Y", P_RH_ITMY, P_RH_ETMY)

    llo = set_cold_state_beam(llo, w0, z, update_factory=None, waist_params=waist_params, losses=losses, is_astig=is_astig)

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
        lock_steps=100,
        gain_scale=0.4,
        pseudo_lock_arms=False,
        run_locks=True
    )
    sol = llo.run(lock)

    runsols = record_sol([0], runsols, include_sol_at=[0], outs=[sol['after locks']])
    runsols = record_model_params([0], [llo], runsols, model_output_params, include_sol_at=[0])

    return llo, sol['after locks'], runsols

def sample_param_variations(params, N):
    """Fills out the 'vals' field of the params dict with uniform random values between min and max"""
    for k in params:
        params[k]['vals'] = np.random.uniform(-params[k]['var'], params[k]['var'], size=(N,)) + params[k]['offset']

def run_population(param_variations,
                   model_output_params,
                   run_thermal=False,
                   idstart=0,
                   idend=10,
                   repopath=str(finesse.ligo.git_path()),
                   is_astig=True,
                   reward_params={},
                   use_real_data=False,
                   new_bs=True,
                   t_evol=5400,
                   **kwargs):

    # Load in the parameter file to make the
    if new_bs:
        factory = ALIGOBeamSplitterFactory(f"{repopath}/LHO/yaml/lho_O4.yaml")
    else:
        factory = ALIGOFactory(f"{repopath}/LHO/yaml/lho_O4.yaml")

    factory.update_parameters(f"{repopath}/LHO/yaml/lho_addRH.yaml")
    factory.params.INPUT.LASER.power = 2

    # Make the model
    factory.reset() # always reset to default
    factory.options.LSC.add_locks = True
    factory.options.LSC.add_output_detectors = True
    factory.options.ASC.add = False
    factory.options.INPUT.add_IMC_and_IM1 = False
    factory.options.thermal.add = True

    runsols = {}
    runsols_noRH = {}
    t0 = time.time()

    include_noRH = any(["_noRH" in k for k in reward_params])

    for i in range(idstart, idend):
        print(f"\nRunning simulation index {i}")
        if i == idstart:
            print_params(param_variations, i)

        # apply the variations
        factory, waist_params, losses, TM_absorptions, RH_efficiency = apply_param_variations(factory, param_variations, i)

        # run the model with nominal RH control
        if run_thermal:
            t, _, _, _, _, runsols = run_thermal_model(factory=factory,
                                                    waist_params=waist_params,
                                                    losses=losses,
                                                    tm_absorption=TM_absorptions,
                                                    return_data_level=2,
                                                    RH_efficiency=RH_efficiency,
                                                    is_astig=is_astig,
                                                    model_output_params=model_output_params,
                                                    runsols=runsols,
                                                    reward_params=reward_params,
                                                    use_real_data=use_real_data,
                                                    repopath=repopath,
                                                    t_evol=t_evol,
                                                    **kwargs)
            if include_noRH:
                # run again with RH control set to 0
                _, _, _, _, _, runsols_noRH = run_thermal_model(factory=factory,
                                                        waist_params=waist_params,
                                                        losses=losses,
                                                        tm_absorption=TM_absorptions,
                                                        return_data_level=2,
                                                        RH_efficiency={"ITMX": 0, "ITMY": 0, "ETMX": 0, "ETMY": 0},
                                                        is_astig=is_astig,
                                                        model_output_params=model_output_params,
                                                        runsols=runsols_noRH,
                                                        reward_params=reward_params,
                                                        use_real_data=use_real_data,
                                                        repopath=repopath,
                                                        t_evol=60,
                                                        **kwargs)
        else:
            _, _, runsols = runligo(factory,
                                    RH_efficiency=RH_efficiency,
                                    waist_params=waist_params,
                                    losses=losses,
                                    is_astig=is_astig,
                                    model_output_params=model_output_params,
                                    runsols=runsols,
                                    **kwargs)

        # undo the variations
        factory = undo_param_variations(factory, param_variations, i)

        if i % 10 == 9:
            print(f"Completed {i+1}/{idend-idstart} simulations in {time.time() - t0:.1f} seconds.\n")

    for k in reward_params:
        if "_noRH" in k and k not in runsols:
            runsols[k] = runsols_noRH[k.strip("_noRH")]

    rewards = eval_population_reward(runsols, reward_params, idend-idstart)

    return runsols, rewards

def set_cold_state_beam(llo, w0, z, update_factory=None, waist_params={}, losses={}, is_astig=False):

    # assert the updating is either through update_factory or w0, z.
    assert (update_factory is not None) ^ (bool(waist_params) and bool(losses)), "Provide either factory or (waist_params, losses)"

    wkey, zkey, astigParams = get_astig_params(w0, z, is_astig)

    # Set Gauss mode wrt PRMAR
    try:
        llo.remove(llo.gIM2_p1_i)
    except Exception as e:
        print(f"An error occurred: {e}\nEnsure factory.options.INPUT.add_IMC_and_IM1 = False")

    llo.add(
        finesse.components.Gauss(
            'g1',
            llo.PRMAR.p2.i,
            **astigParams,
        )
    )

    # if update_factory
    if update_factory is not None:
        # only update beam params if update_factory is provided.
        for k in llo.elements['g1'].parameters:
            if 'beam_waist' in update_factory.params:
                if (wkey in k.name) or (zkey in k.name):
                    print(f"Updating {k.name} with {update_factory.params.beam_waist[k.name]}")
                    k.value = update_factory.params.beam_waist[k.name]
        # No need to update losses as they are already set in the update_factory.

    # if not update_factory
    elif waist_params and losses:
        # beam params
        for k in llo.elements['g1'].parameters:
            if (wkey in k.name) or (zkey in k.name):
                print(f"Updating {k.name} with {waist_params[k.name]}")
                k.value = waist_params[k.name]

        # losses
        llo.X_arm_loss = (losses['carm'] + losses['darm']) * 1e-6
        llo.Y_arm_loss = (losses['carm'] - losses['darm']) * 1e-6
        print(f"Updating losses: XARM: {llo.X_arm_loss}, YARM: {llo.Y_arm_loss}")

    return llo

def llo_O4_HR_baffle(r_lim=0.21, N=200, AoI=45):
    x = np.linspace(-r_lim, r_lim, N)
    X, Y = np.meshgrid(x, x)
    inches_2_m = 25.4e-3
    r_major = 11.69 / 2 * inches_2_m * np.cos(np.deg2rad(AoI))
    r_minor = 10.236 / 2 * inches_2_m
    x_offset = 1.433 * inches_2_m * np.cos(np.deg2rad(AoI))

    # in the coordinates of BS
    ellipse_main = ((X - x_offset + 37.5e-3) / r_major) ** 2 + (Y / r_minor) ** 2
    ellipse_offset = ((X + x_offset + 37.5e-3) / r_major) ** 2 + (Y / r_minor) ** 2
    bs_hr_baffle = np.ones_like(X)
    bs_hr_baffle[np.logical_and(ellipse_main > 1, ellipse_offset > 1)] = 0

    return x, bs_hr_baffle

def llo_O4_BS_AR_baffle(r_lim=0.21, N=200, AoI=45, offset_direction=-1):
    x  = np.linspace(-r_lim, r_lim, N)
    X, Y = np.meshgrid(x, x)
    inches_2_m = 25.4e-3

    # Use Snell law to compute offset from input
    xoffset_out = offset_direction * (80e-3 - (60e-3 * np.tan(np.arcsin((1/np.sqrt(2))/1.4996))))
    x_offset = offset_direction * 2.74 * inches_2_m * np.cos(np.deg2rad(AoI))
    r_major = 11.69 / 2 * inches_2_m * np.cos(np.deg2rad(AoI))
    r_minor = 10.236 / 2 * inches_2_m
    ellipse_main = ((X - x_offset + xoffset_out)/ r_major) ** 2 + (Y / r_minor) ** 2
    ellipse_offset = ((X + x_offset + xoffset_out)/ r_major) ** 2 + (Y / r_minor) ** 2
    bs_ar_baffle = np.ones_like(X)
    bs_ar_baffle[np.logical_and(ellipse_main > 1, ellipse_offset > 1)] = 0
    return x, bs_ar_baffle

def llo_O4_BS_HR_aperture(r_lim = 0.185, N=251, AoI = 45):
    x  = np.linspace(-r_lim, r_lim, N)
    X, Y = np.meshgrid(x, x)
    r_major = r_lim 
    r_minor = r_lim * np.cos(np.deg2rad(AoI))
    bs_ellipse= (X / r_minor) ** 2 + (Y / r_major) ** 2
    bs_aperture = np.ones_like(X)
    bs_aperture[bs_ellipse > 1] = 0
    return x, bs_aperture

def llo_O4_BS_AR_aperture(r_lim = 0.24, N=301, 
                          AoI=np.arcsin(1/np.sqrt(2)/FusedSilica.nr),
                          offset = 1):
    x  = np.linspace(-r_lim, r_lim, N)
    X, Y = np.meshgrid(x, x)
    a_bs = 0.185
    r_major = a_bs 
    r_minor = a_bs * np.cos(np.deg2rad(AoI))
    t_bs = 60e-3
    x_offset = offset * t_bs * np.tan(AoI)
    bs_ellipse= ((X-x_offset) / r_minor) ** 2 + (Y / r_major) ** 2
    bs_aperture = np.ones_like(X)
    bs_aperture[bs_ellipse > 1] = 0
    return x, bs_aperture

def set_ringheaters(llo, factory, arm, P_RH_ITM, P_RH_ETM):
    lens = llo.get(f"ITM{arm}lens")
    lens.f = 1 / (1 / lens.f + P_RH_ITM * factory.params.IRH_sub)
    itm = llo.get(f"ITM{arm}")
    etm = llo.get(f"ETM{arm}")
    itm.Rc = 2 / (2 / itm.Rc + P_RH_ITM * factory.params.IRH_srf)
    etm.Rc = 2 / (2 / etm.Rc + P_RH_ETM * factory.params.ERH_srf)
    print(f"New values for ITM{arm} lens f: {lens.f.eval()}, ITM{arm} Rc: {itm.Rcx.eval()}, ETM{arm} Rc: {etm.Rcx.eval()}")

def run_thermal_model(factory=None,
                      maxtems=8,
                      datapath=None,
                      return_data_level=0,
                      update_yaml=None,
                      w0=1.015e-3,
                      z=6.0,
                      RH_efficiency={'ITMX': 0.9, 'ITMY': 0.9, 'ETMX': 0.9, 'ETMY': 0.9},
                      t_evol=10000,
                      tm_absorption={'ITMX': 0.5e-6, 'ITMY': 0.5e-6, 'ETMX': 0.3e-6, 'ETMY': 0.3e-6},
                      waist_params={},
                      losses={},
                      is_astig=True,
                      model_output_params=None,
                      runsols=None,
                      reward_params={},
                      use_real_data=False,
                      repopath=None,
                      rec_therm_lenses=False,
                      new_bs=True,
                      add_BS_baffle=True,
                      initial_data_time=1418514918,
                      override_RHeff=False,
                      addl_parse=None,
                      **kwargs):
    """
    Run the thermal model for the LIGO interferometer.

    Parameters:
    factory: ALIGOFactory object
        Factory object to use for the model. If None, a new factory is created.
    maxtems: int
        Maximum TEM mode to consider in the model.
    datapath: str
        Path to save the data to. Default is None.
    return_data_level: int
        Level of data to return. 0: None, 1: Data for GA rewards, 2: All data.
    update_yaml: str
        Path to the yaml file to update the factory with.
    w0: float
        Beam waist size if update_yaml is being used. Default is 1.015e-3.
    z: float
        Rayleigh range if update_yaml is being used. Default is 6.0.
    RH_efficiency: float
        Ring heater efficiency. Default is {'ITMX': 0.9, 'ITMY': 0.9, 'ETMX': 0.9, 'ETMY': 0.9}.
    t_evol: int
        Time to evolve the model for. Default is 10000.
    tm_absorption: dict
        Absorption of the test masses as a dict {'ITMX': 0.3e-6, ...}. Default is {'ITMX': 0.5e-6, 'ITMY': 0.5e-6, 'ETMX': 0.3e-6, 'ETMY': 0.3e-6}.
    waist_params: dict
        Waist parameters to update the factory with. Default is {}.
    losses: dict
        Losses to update the factory with. Default is {}.
    is_astig: bool
        Whether the beam is astigmatic. Default is True.
    model_output_params: list
        List of model parameters to record. Default is None.
    runsols: dict
        Dictionary to store the results in. Default is None.
    use_real_data: bool
        Whether to use real data for input power. Default is False.
    repopath: str
        Path to the ligo-commissionin-modeling. Default is None.
    rec_therm_lenses: bool
        Whether to record the focal lengths of thermal lenses. Default is False.
    new_bs: bool
        Whether to use the new beam splitter model. Default is True.
    add_BS_baffle: bool
        Whether to add the baffle to the beam splitter. Default is True.
    initial_data_time: int
        Start time for the real power up data that is used if use_real_data is True. Default is 1418514918.
    override_RHeff: bool
        if True, ignore yaml input for RH efficiency and override with kwarg input.
    addl_parse: string
        additional kat script to be parsed for the model. Default is None.
    kwargs: Additional keyword arguments
    """

    if not factory:
        if new_bs:
            factory = ALIGOBeamSplitterFactory(f"{repopath}/LHO/yaml/lho_O4.yaml")
        else:
            factory = ALIGOFactory(f"{repopath}/LHO/yaml/lho_O4.yaml")
        factory.update_parameters(f"{repopath}/LHO/yaml/lho_addRH.yaml")

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
    factory.options.apertures.add = False       
    factory.options.apertures.test_mass = True
    factory.options.apertures.PR3_SR3 = True
    factory.options.apertures.use_surface_profile = True
    factory.options.apertures.BS_ITM=False  

    llo = factory.make()


    llo.alpha_ITMX = factory.params.absorption.ITMX
    llo.alpha_ITMY = factory.params.absorption.ITMY
    llo.alpha_ETMX = factory.params.absorption.ETMX
    llo.alpha_ETMY = factory.params.absorption.ETMY
    llo.IRH_sub=factory.params.IRH_sub
    llo.IRH_srf=factory.params.IRH_srf
    llo.ERH_srf=factory.params.ERH_srf
    llo.ERH_sub=factory.params.ERH_sub

    llo.remove(llo.gIM2_p1_i)
    llo.add(
        finesse.components.Gauss(
            'gIM2_p1_i',
            llo.PRMAR.p2.i,
            w0x=factory.params.beam_waist.w0x,
            zx=factory.params.beam_waist.zx,
            w0y=factory.params.beam_waist.w0y,
            zy=factory.params.beam_waist.zy))



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

    if update_yaml:
        llo = set_cold_state_beam(llo, w0, z, update_factory=factory, waist_params={}, losses={}, is_astig=is_astig)
    else:
        llo = set_cold_state_beam(llo, w0, z, update_factory=None, waist_params=waist_params, losses=losses, is_astig=is_astig)

    # make the thermal model
    ts_itmx, ts_etmx, ts_itmy, ts_etmy = make_ts_optics()
    add_thermal_detectors(llo)

    # override RH_efficiency if present in factory params
    if 'RH_eff' in factory.params and not override_RHeff:
        for TMkey in factory.params.RH_eff:
            RH_efficiency[TMkey] = factory.params.RH_eff[TMkey]

    # ring heater powers [W] with 70% efficiency from requested power to optic
    P_RH_ITMX = 2*factory.params.P_RH_ITMX * RH_efficiency['ITMX']
    P_RH_ITMY = 2*factory.params.P_RH_ITMY * RH_efficiency['ITMY']
    P_RH_ETMX = 2*factory.params.P_RH_ETMX * RH_efficiency['ETMX']
    P_RH_ETMY = 2*factory.params.P_RH_ETMY * RH_efficiency['ETMY']

    llo.ITMXlens.f = 1 / (1 / llo.ITMXlens.f + 1/(225e3)) 
    llo.ITMYlens.f = 1 / (1 / llo.ITMYlens.f + 1/(-247e3)) 

    set_ringheaters(llo, factory, "X", P_RH_ITMX, P_RH_ETMX)
    set_ringheaters(llo, factory, "Y", P_RH_ITMY, P_RH_ETMY)

    base = llo.deepcopy()
    initial_params = {p.full_name: p.value for p in base.all_parameters}

    # Run cold state model to set starting point
    R = 0.17
    N = 201

    x, TM_aperture = aligo_O4_TM_aperture(R, N)
    x, lens_aperture = aligo_O4_ESD_inner_aperture(R, N)
    y = x

    # Get surfaces
    for TM in [llo.ITMX, llo.ITMY, llo.ETMX, llo.ETMY]:
        if 'X' in TM.name:
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
        lock_steps=100,
        gain_scale=0.4,
        pseudo_lock_arms=False,
        run_locks=True
    )

    sol = llo.run(lock)
    sol1 = sol['after locks']
    llo_drmi = llo.deepcopy()

    # save initial gains to be changed during power up
    initial_gains = {}
    for lsc_dof in ['DARM_rf', 'CARM', 'PRCL', 'SRCL', 'MICH']:
        initial_gains[lsc_dof] = llo.get(f'{lsc_dof}_lock.gain')

    # update thermal state with rom:
    print(f"Updating absorption values with {tm_absorption}")
    itmx_absorption = tm_absorption['ITMX']
    itmy_absorption = tm_absorption['ITMY']
    etmx_absorption = tm_absorption['ETMX']
    etmy_absorption = tm_absorption['ETMY']

    def get_data(chan_file: str = 'data.pkl'):
        '''
        To use real P_in data, in a terminal, first need to run
        >> python ligo-commissioning-modeling/scripts/data_munch.py L1_chans.yaml data.pkl
        for list of channels and times specified in L1_chans.yaml outputted to data.pkl
        can change the time interval in the L1_chans.yaml file
        '''
        with open(chan_file, 'rb') as f:
            data=pickle.load(f)
            return data

    if use_real_data:
        print(f'Using real P_in data from IFO powerup at {initial_data_time}')
        d = get_data(f"{repopath}/scripts/data_{initial_data_time}.pkl")
        Power_data = d[initial_data_time]['data']['H1:IMC-IM4_TRANS_NSUM_OUT16'].value
        Power_data_dt = d[initial_data_time]['data']['H1:IMC-IM4_TRANS_NSUM_OUT16'].dt.value

    def update_ts(base1, L0Pmax=64, use_real_data=False):

        if ts_itmx.t == 0:
            llo.ITMYlens.f.value = base1.ITMYlens.f.value
            llo.ITMXlens.f.value = base1.ITMXlens.f.value

        if use_real_data:
            try:
                current_P = Power_data[int(ts_itmx.t/Power_data_dt)]
                llo.L0.P = current_P
            except IndexError:
                current_P = llo.L0.P # keep using last known P_in if data has ended.
            print('Current time is {}s, power in is {}W.'.format(ts_itmx.t,current_P))

            # update gains
            for lsc_dof in ['DARM_rf', 'CARM', 'PRCL', 'SRCL', 'MICH']:
                obj = getattr(llo, f'{lsc_dof}_lock')
                obj.gain = initial_gains[lsc_dof] * initial_P / current_P

        else:
            if ts_itmx.t > 180 and llo.L0.P != 25 and llo.L0.P < 25:
                    llo.L0.P = 25
                    llo.DARM_rf_lock.gain *= 2 / 25
                    llo.CARM_lock.gain /= 25 / 2
                    llo.PRCL_lock.gain /= 25 / 2
                    llo.SRCL_lock.gain /= 25 / 2
                    llo.MICH_lock.gain /= 25 / 2

            elif ts_itmx.t > 180 + 10 * 64 and llo.L0.P != 64:
                    llo.L0.P = L0Pmax
                    llo.CARM_lock.gain /= 64 / 25
                    llo.DARM_rf_lock.gain /= 64 / 25
                    llo.PRCL_lock.gain /= 64 / 25
                    llo.SRCL_lock.gain /= 64 / 25
                    llo.MICH_lock.gain /= 64 / 25

        # Update intensity:
        if ts_itmx.t < 1000:
            ts_itmx.dt = ts_etmx.dt = 20
            ts_itmy.dt = ts_etmy.dt = 20
        elif ts_itmx.t < 1500:
            ts_itmx.dt = ts_etmx.dt = 100
            ts_itmy.dt = ts_etmy.dt = 100
        elif ts_itmx.t >= 1500:
            ts_itmx.dt = ts_etmx.dt = 200
            ts_itmy.dt = ts_etmy.dt = 200

        for hr_func, _ts_optic, _abs in zip([I_ITMX_HR, I_ETMX_HR, I_ITMY_HR, I_ETMY_HR],
                                            [ts_itmx, ts_etmx, ts_itmy, ts_etmy],
                                            [itmx_absorption, etmx_absorption, itmy_absorption, etmy_absorption]):
            u_k, I_abs = hr_func(llo, values, absorption =  _abs)
            _ts_optic.I.append(I_abs)
            _ts_optic.uI.append(u_k)
            compute_new_opd_state(_ts_optic)
            compute_new_deformation_state(_ts_optic)
            _ts_optic.t += _ts_optic.dt

    initial_P = 2
    llo.L0.P = initial_P
    values = SimpleNamespace()
    values.out = None
    values.x = x
    values.y = y
    t = [0]
    models = [llo.deepcopy()]
    llo.beam_trace()
    outs = [llo.run()]
    values.out = outs[0]
    locks = []
    extra_outs = {
                'ITMXlens': [],
                'ITMYlens': [],
                'ITMX_Rc': [],
                'ITMY_Rc': [],
                'ETMX_Rc': [],
                'ETMY_Rc': [],
                }

    program_starts = time.time()
    while ts_itmx.t <= t_evol:
        print(ts_itmx.t)
        update_ts(base, use_real_data=use_real_data, **kwargs)
        t.append(ts_itmx.t)

        update_maps(initial_params, llo, values, ts_itmx, ts_etmx, ts_itmy, ts_etmy)
        llo.run(fac.SetLockGains(gain_scale=0.4))

        if return_data_level == 2:
            llo_hot = llo.deepcopy()
            models.append(llo_hot)

        sols = llo.run("series(run_locks(exception_on_fail=False, max_iterations=500), noxaxis())")

        if return_data_level == 2:
            locks.append(sols['run locks'])

        values.out = sols["noxaxis"]
        outs.append(values.out)

        if rec_therm_lenses:
            extra_outs['ITMXlens'].append(llo.ITMXlens.f.value)
            extra_outs['ITMYlens'].append(llo.ITMYlens.f.value)
            if return_data_level == 2:
                extra_outs['ITMX_Rc'].append(llo_hot.ITMX.Rcx.value)
                extra_outs['ITMY_Rc'].append(llo_hot.ITMY.Rcx.value)
                extra_outs['ETMX_Rc'].append(llo_hot.ETMX.Rcx.value)
                extra_outs['ETMY_Rc'].append(llo_hot.ETMY.Rcx.value)

        print(ts_itmx.t, sols["noxaxis"]["Parm"], sols["noxaxis"]["PRG"], llo.ITMXlens.f.value, llo.ITMYlens.f.value)

        if np.isnan(sols["noxaxis"]["Parm"]) or np.isnan(sols["noxaxis"]["PRG"]):
            # just replacing with a dummy solution with outputs 0
            sol_outputs = sols["noxaxis"].outputs
            sols = {"noxaxis": {}}
            for k in sol_outputs:
                sols["noxaxis"][k] = 0
            break

    end_times = time.time()
    print(f'Elapsed time: {end_times - program_starts} s')

    if datapath is not None:
        print(f'saving data to {datapath}')
        with open(datapath, "wb") as file:
            pickle.dump(
                {
                    "parameters": factory.params,
                    "options": factory.options,
                    "outs": outs,
                    "t": t,
                    # "models": [m.unparse() for m in models],
                },
                file,
            )

    if runsols is not None:
        k1 = list(reward_params.keys())[0]
        runsols = record_sol(t, runsols, include_sol_at=reward_params[k1]['timestamps'], outs=outs)
        runsols = record_model_params(t, models, runsols, model_output_params, include_sol_at=reward_params[k1]['timestamps'])

    if return_data_level == 1:
        return t, llo_drmi, sol1, llo, sols["noxaxis"], runsols

    if return_data_level == 2:
        return t, outs, models, locks, extra_outs, runsols

    del models, locks, outs, values

def plot_thermal_evolution(fulldata,
                           control,
                           plotfileloc='llo_power_up_dump.pdf',
                           axislims=False,
                           colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                           ):

    with PdfPages(plotfileloc) as pdf:
        plt.figure(figsize=(8, 5))
        for j, data in enumerate(fulldata):
            plt.plot(data["t"], tuple(out['Pin'] for out in data["outs"]), label=f"{control['param']}: {data['varyParam']}", color=colors[j])
        plt.ylabel("Power [W]")
        plt.xlabel("Time [s]")
        plt.title("Input power")
        plt.legend()
        if axislims: plt.ylim(0, 4)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(8, 5))
        for j, data in enumerate(fulldata):
            plt.plot(data["t"], tuple(out['PreflPRM'] for out in data["outs"]), label=f"{control['param']}: {data['varyParam']}", color=colors[j])
        plt.ylabel("Power [W]")
        plt.xlabel("Time [s]")
        plt.title("REFL")
        plt.legend()
        if axislims: plt.ylim(0, 5)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(8, 5))
        for j, data in enumerate(fulldata):
            c = colors[j]
            plt.plot(data["t"], tuple(out['Px']/1e3 for out in data["outs"]), label=f"X arm; {control['param']}: {data['varyParam']}", color=c)
            plt.plot(data["t"], tuple(out['Py']/1e3 for out in data["outs"]), label=f"Y arm; {control['param']}: {data['varyParam']}", ls="--", color=c)
        plt.ylabel("Power [kW]")
        plt.xlabel("Time [s]")
        plt.title("Arm power")
        if axislims: plt.ylim(10, 12)
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(8, 5))
        for j, data in enumerate(fulldata):
            c = colors[j]
            plt.plot(data["t"], tuple(out['AGX'] for out in data["outs"]), label=f"X arm; {control['param']}: {data['varyParam']}", ls='-', color=c)
            plt.plot(data["t"], tuple(out['AGY'] for out in data["outs"]), label=f"Y arm; {control['param']}: {data['varyParam']}", ls='--', color=c)
        plt.ylabel("Gain")
        plt.xlabel("Time [s]")
        plt.title("Arm gains")
        plt.legend()
        if axislims: plt.ylim(260, 265)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # plt.figure(figsize=(8, 5))
        # for j, data in enumerate(fulldata):
        #     c = colors[j]
        #     plt.plot(data["t"], tuple(out['PRG9'] for out in data["outs"]), label=f"9; {control['param']}: {data['varyParam']}", ls='-', color=c)
        #     plt.plot(data["t"], tuple(out['PRG45'] for out in data["outs"]), label=f"45; {control['param']}: {data['varyParam']}", ls='--', color=c)
        #     plt.plot(data["t"], tuple(out['PRG'] for out in data["outs"]), label=f"Carrier; {control['param']}: {data['varyParam']}", ls='-.', color=c)
        # plt.ylabel("Gain")
        # plt.xlabel("Time [s]")
        # plt.title("Recycling gains")
        # plt.legend()
        # if axislims: plt.ylim(0, 130)
        # plt.tight_layout()
        # pdf.savefig()
        # plt.close()

        plt.figure(figsize=(8, 5))
        for j, data in enumerate(fulldata):
            plt.plot(data["t"], tuple(out['PRG'] for out in data["outs"]), label=f"Carrier; {control['param']}: {data['varyParam']}", color=colors[j])
        plt.ylabel("Gain")
        plt.xlabel("Time [s]")
        plt.title("Power Recycling Gain")
        plt.legend()
        if axislims: plt.ylim(43, 44)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(8, 5))
        for j, data in enumerate(fulldata):
            # plt.plot(data["t"], tuple(np.sum(out['E_prc_u9'] * out['E_prc_l9'].conjugate() + out['E_prc_u9'].conjugate() * out['E_prc_l9']).real/out['Pin'] for out in data["outs"]), label=f"{control['param']}: {data['varyParam']}", color=colors[j])
            plt.plot(data["t"], tuple(out['PRG9'] for out in data["outs"]), label=f"9; {control['param']}: {data['varyParam']}", color=colors[j])
        plt.ylabel("Gain")
        plt.xlabel("Time [s]")
        # plt.title("RF18 / P_IN")
        plt.title("PRG9")
        plt.legend()
        if axislims: plt.ylim(0.645, 0.657)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(8, 5))
        for j, data in enumerate(fulldata):
            # plt.plot(data["t"], tuple(np.sum(out['E_prc_u45'] * out['E_prc_l45'].conjugate() + out['E_prc_u45'].conjugate() * out['E_prc_l45']).real/out['Pin'] for out in data["outs"]), label=f"{control['param']}: {data['varyParam']}", color=colors[j])
            plt.plot(data["t"], tuple(out['PRG45'] for out in data["outs"]), label=f"45; {control['param']}: {data['varyParam']}", color=colors[j])
        plt.ylabel("Gain")
        plt.xlabel("Time [s]")
        # plt.title("RF90 / P_IN")
        plt.title("PRG45")
        plt.legend()
        if axislims: plt.ylim(0.101, 0.102)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

def generate_plots(plotloc,
                   initial_data_time,
                   outs,
                   irange1,
                   t,
                   t_data,
                   P_in,
                   P_refl,
                   P_pop,
                   Kappa_C,
                   P_PRG,
                   RF18,
                   RF90,
                   ylim,
                   tollow,
                   tolhigh,
                   maxx,
                   factory,
                   savefig=True,
                  ):

    lambda_op = lambda x, y, op: op(op(x), op(y))

    with PdfPages(plotloc) as pdf:
        # First page with the first plot
        fig, ax1 = plt.subplots(figsize=(11, 8.5))  # Landscape orientation
        plt.title("Input power")
        color = 'tab:blue'
        ax1.set_xlabel(f'Time after {initial_data_time} [s]')
        ax1.set_ylabel('Power [W]', color=color)

        simPin = np.array([out['Pin'] for out in outs])[irange1]
        miny, maxy = lambda_op(simPin, P_in, min), lambda_op(simPin, P_in, max)
        ax1.plot(t[irange1], simPin, color=color)
        if ylim: ax1.set_ylim(miny*tollow, maxy*tolhigh)
        ax1.set_xlim(-100, maxx)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('H1:IMC-IM4_TRANS_NSUM_OUT16', color=color)
        ax2.plot(t_data, P_in, label='data', color=color, alpha=0.6)
        if ylim: ax2.set_ylim(miny*tollow, maxy*tolhigh)
        ax2.tick_params(axis='y', labelcolor=color)

        if savefig:
            pdf.savefig()
            plt.close()
        else:
            plt.show()

        # Second page with the next six plots in a grid
        fig, axs = plt.subplots(2, 3, figsize=(18, 8.5))  # 2x3 grid, landscape orientation

        # Plot 2: REFL
        ax1 = axs[0, 0]
        ax1.set_title("REFL")
        color = 'tab:blue'
        ax1.set_xlabel(f'Time after {initial_data_time} [s]')
        ax1.set_ylabel('Power [W]', color=color)

        simPreflPRM = np.array([out['PreflPRM'] for out in outs])[irange1]
        miny, maxy = lambda_op(simPreflPRM, P_refl, min), lambda_op(simPreflPRM, P_refl, max)
        ax1.plot(t[irange1], simPreflPRM, color=color)
        if ylim: ax1.set_ylim(miny*tollow, maxy*tolhigh)
        ax1.set_xlim(-100, maxx)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('L1:LSC-REFL_A_LF_OUTPUT', color=color)
        ax2.plot(t_data, P_refl, label='data', color=color, alpha=0.6)
        if ylim: ax2.set_ylim(miny*tollow, maxy*tolhigh)
        ax2.tick_params(axis='y', labelcolor=color)

        # Plot 3: Pick-off PRC Power
        ax1 = axs[0, 1]
        ax1.set_title("Pick-off PRC Power")
        color = 'tab:blue'
        ax1.set_xlabel(f'Time after {initial_data_time} [s]')
        ax1.set_ylabel('Power [W]', color=color)
        simPpop = np.array([out['Ppop']/factory.params.PRC.PR2.T for out in outs])[irange1]
        miny, maxy = lambda_op(simPpop, P_pop, min), lambda_op(simPpop, P_pop, max)
        ax1.plot(t[irange1], simPpop, color=color)
        if ylim: ax1.set_ylim(miny*tollow, maxy*tolhigh)
        ax1.set_xlim(-100, maxx)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('L1:LSC-POP_A_LF_OUTPUT', color=color)
        ax2.plot(t_data, P_pop, label='data', color=color, alpha=0.6)
        if ylim: ax2.set_ylim(miny*tollow, maxy*tolhigh)
        ax2.tick_params(axis='y', labelcolor=color)

        # Plot 4: Arm gains
        ax1 = axs[0, 2]
        ax1.set_title('Arm gains')
        color = 'tab:blue'
        ax1.set_xlabel(f'Time after {initial_data_time} [s]')
        ax1.set_ylabel('Gain', color=color)

        simAGX = np.array([out['AGX'] for out in outs])[irange1]
        miny, maxy = lambda_op(simAGX, Kappa_C, min), lambda_op(simAGX, Kappa_C, max)
        ax1.plot(t[irange1], simAGX, color=color, label='X arm')
        ax1.set_xlim(-100, maxx)

        simAGY = np.array([out['AGY'] for out in outs])[irange1]
        ax1.plot(t[irange1], simAGY, color='orange', label='Y arm')
        ax1.set_xlim(-100, maxx)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend()

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('L1:CAL-CS_TDEP_KAPPA_C_OUTPUT', color=color)
        ax2.plot(t_data, Kappa_C, label='data', color=color, alpha=0.6)
        ax2.tick_params(axis='y', labelcolor=color)

        # Plot 5: Recycling gains
        ax1 = axs[1, 0]
        ax1.set_title('Recycling gains')
        color = 'tab:blue'
        ax1.set_xlabel(f'Time after {initial_data_time} [s]')
        ax1.set_ylabel('Gain', color=color)
        simPRG = np.array([out['PRG'] for out in outs])[irange1]
        miny, maxy = lambda_op(simPRG, P_PRG, min), lambda_op(simPRG, P_PRG, max)
        ax1.plot(t[irange1], simPRG, color=color)
        if ylim: ax1.set_ylim(miny*tollow, maxy*tolhigh)
        ax1.set_xlim(-100, maxx)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('L1:LSC-PRC_GAIN_MON', color=color)
        ax2.plot(t_data, P_PRG, label='data', color=color, alpha=0.6)
        if ylim: ax2.set_ylim(miny*tollow, maxy*tolhigh)
        ax2.tick_params(axis='y', labelcolor=color)

        # Plot 6: PRG9
        ax1 = axs[1, 1]
        ax1.set_title("PRG9")
        color = 'tab:blue'
        ax1.set_xlabel(f'Time after {initial_data_time} [s]')
        ax1.set_ylabel('Power [arb.]', color=color)
        simPRG9 = np.array([out['PRG9'] for out in outs])[irange1]
        miny, maxy = lambda_op(simPRG9, RF18, min), lambda_op(simPRG9, RF18, max)
        ax1.plot(t[irange1], simPRG9, color=color)
        if ylim: ax1.set_ylim(miny*tollow, maxy*tolhigh)
        ax1.set_xlim(-100, maxx)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('POPAIR_B_RF18_I_MON / P_in', color=color)
        ax2.plot(t_data, RF18, label='data', color=color, alpha=0.6)
        if ylim: ax2.set_ylim(miny*tollow, maxy*tolhigh)
        ax2.tick_params(axis='y', labelcolor=color)

        # Plot 7: PRG45
        ax1 = axs[1, 2]
        ax1.set_title("PRG45")
        color = 'tab:blue'
        ax1.set_xlabel(f'Time after {initial_data_time} [s]')
        ax1.set_ylabel('Power [arb.]', color=color)
        simPRG45 = np.array([out['PRG45'] for out in outs])[irange1]
        miny, maxy = lambda_op(simPRG45, RF90, min), lambda_op(simPRG45, RF90, max)
        ax1.plot(t[irange1], simPRG45, color=color)
        if ylim: ax1.set_ylim(miny*tollow, maxy*tolhigh)
        ax1.set_xlim(-100, maxx)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('POPAIR_B_RF90_I_MON / P_in', color=color)
        ax2.plot(t_data, RF90, label='data', color=color, alpha=0.6)
        if ylim: ax2.set_ylim(miny*tollow, maxy*tolhigh)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.tight_layout()
        if savefig:
            pdf.savefig()
            plt.close()
        else:
            plt.show()

def single_reward(sol, reward_params1, j):

    diff = abs(reward_params1['centre'][j] - sol)

    if diff <= reward_params1['plateu'][j]:
        return reward_params1['weight'][j]
    elif reward_params1['plateu'][j] <= diff:
        return reward_params1['weight'][j] * (reward_params1['base'][j] - diff) / (reward_params1['base'][j] - reward_params1['plateu'][j])
    else:
        return -1e3

def eval_population_reward(sols, reward_params, N):
    rewards = np.zeros(N)
    for i in range(N):
        for k in reward_params:
            for j in range(len(reward_params[k]['type'])):
                if reward_params[k]['type'][j] == 'abs':
                    rewards[i] += single_reward(sols[k][i][j], reward_params[k], j)
                elif reward_params[k]['type'][j] == 'rel':
                    rewards[i] += single_reward(sols[k][i][j]/sols[k][i][0], reward_params[k], j)
    return rewards

def mutation_crossover(param_variations, topN, N, idx, frac_crossover=0.5, frac_mutation=0.8):
    """
    Function to perform crossover (avg of two solutions) and mutation (random variation about itself) on the population of solutions.

    Parameters:
    param_variations: dict of parameter variations
    topN: number of top solutions to keep as parents
    N: total number of solutions
    idx: indices of topN solutions
    frac_crossover: fraction of solutions to crossover
    frac_mutation: fraction of solutions to mutate
    frac_shrink: fraction of mutation size to shrink by

    Returns:
    new_pop: dict of new population of solutions
    """

    # generate new population of solutions (including parents as well);
    new_pop = deepcopy(param_variations)
    for k in param_variations:
        for i in range(topN):
            new_pop[k]['vals'][i] = param_variations[k]['vals'][idx[i]]

    # crossover and mutation
    for i in range(topN, N):
        # Create weights for sampling based on index position
        weights = np.arange(len(idx), 0, -1) + len(idx)/5  # weighting indices (but not going down to 0)
        weights = weights / weights.sum()  # create a probability distribution

        ir = np.random.choice(idx, p=weights)

        # copy parent as it is
        for k in param_variations:
            new_pop[k]['vals'][i] = param_variations[k]['vals'][ir]

        # crossover
        if np.random.rand() < frac_crossover:

            if len(idx) > 1:
                # Ensure ir2 is different from ir
                idx2 = idx[idx != ir]
                weights2 = weights[idx != ir]
                weights2 /= weights2.sum()
                ir2 = np.random.choice(idx2, p=weights2)
            else:
                ir2 = ir

            for k in param_variations:
                new_pop[k]['vals'][i] = (param_variations[k]['vals'][ir] + param_variations[k]['vals'][ir2]) / 2

        # mutation
        if np.random.rand() < frac_mutation:
            for k in param_variations:
                new_pop[k]['vals'][i] += np.random.uniform(-param_variations[k]['var'], param_variations[k]['var'])

    return new_pop

def get_actual_params(all_sols,
                      w0=1.015e-3,
                      z=6.0,
                      repopath='./',
                      is_astig=False,
                      ):

    factory = ALIGOFactory(f"{repopath}/LHO/yaml/lho_O4.yaml")
    factory.update_parameters(f"{repopath}/LHO/yaml/lho_addRH.yaml")
    factory.params.INPUT.LASER.power = 2

    factory.reset()
    data = {}

    wkey, zkey, astigParams = get_astig_params(w0, z, is_astig)

    for ii in range(len(all_sols)):
        _, _, _, newpop = all_sols[ii]['sols'], all_sols[ii]['rewards'], all_sols[ii]['topidx'], all_sols[ii]['newpop']
        data[ii] = {}
        llo = factory.make()
        if newpop['beam_waist/w0x']['type'] == 'abs':
            llo.remove(llo.gIM2_p1_i)
            llo.add(
                finesse.components.Gauss(
                    'g1',
                    llo.PRMAR.p2.i,
                    **astigParams,
                )
            )

        for path, item in newpop.items():
            p = path.split('/')

            if p[0] == 'beam_waist':
                if newpop['beam_waist/w0x']['type'] == 'rel':
                    for k in llo.elements['gIM2_p1_i'].parameters:
                        if k.name == p[1]:
                            if (wkey in k.name) or (zkey in k.name):
                                data[ii][path] = item['vals']
                else:
                    for k in llo.elements['g1'].parameters:
                        if k.name == p[1]:
                            if (wkey in k.name) or (zkey in k.name):
                                data[ii][path] = item['vals']
                continue

            elif p[0] in ('loss', 'absorption', 'RH_eff'):
                data[ii][path] = item['vals']
                continue

            if len(p) == 2:
                if newpop[path]['type'] == 'abs':
                    data[ii][path] = factory.params[p[0]][p[1]] + item['vals']
                elif newpop[path]['type'] == 'rel':
                    data[ii][path] = factory.params[p[0]][p[1]] * (1 + item['vals'])
            elif len(p) == 3:
                if newpop[path]['type'] == 'abs':
                    data[ii][path] = factory.params[p[0]][p[1]][p[2]] + item['vals']
                elif newpop[path]['type'] == 'rel':
                    data[ii][path] = factory.params[p[0]][p[1]][p[2]] * (1 + item['vals'])

    # get min and max values for each parameter
    minmax = {}
    for k in data[0]:
        minmax[k] = [data[0][k].min(), data[0][k].max()]
        for ii in range(1, len(data)):
            minmax[k][0] = min(minmax[k][0], data[ii][k].min())
            minmax[k][1] = max(minmax[k][1], data[ii][k].max())

    return data, minmax

def get_nested_dict(lastgendata):
    nesteddict = {}
    for path in lastgendata:
        p = path.split('/')
        if len(p) == 1:
            nesteddict[p[0]] = lastgendata[path]
        elif len(p) == 2:
            if p[0] not in nesteddict:
                nesteddict[p[0]] = {}
            nesteddict[p[0]][p[1]] = lastgendata[path]
        elif len(p) == 3:
            if p[0] not in nesteddict:
                nesteddict[p[0]] = {}
            if p[1] not in nesteddict[p[0]]:
                nesteddict[p[0]][p[1]] = {}
            nesteddict[p[0]][p[1]][p[2]] = lastgendata[path]

    if 'loss' in nesteddict:
        if 'carm' in nesteddict['loss'] and 'darm' in nesteddict['loss']:
            nesteddict['X'] = {'arm_loss': (nesteddict['loss']['carm'] + nesteddict['loss']['darm'])*1e-6}
            nesteddict['Y'] = {'arm_loss': (nesteddict['loss']['carm'] - nesteddict['loss']['darm'])*1e-6}
        if 'PRCL' in nesteddict['loss']:
            nesteddict['PRC']['PR3']['L'] = nesteddict['loss']['PRCL']*1e-6

        del nesteddict['loss']

    return nesteddict

# Function to recursively print the nested dictionary
def nested_dict_yamlout(d, index, indent=0, yamltext=''):
    for k, v in d.items():
        if isinstance(v, dict):
            yamltext += ' ' * indent + f'{k}:\n'
            yamltext = nested_dict_yamlout(v, index, indent+4, yamltext)
        else:
            yamltext += ' ' * indent + f'{k}: {v[index]}\n'

    return yamltext

# function to write yaml file for given index
def write_yaml_file(d, index, filename, print_text=True, **kwargs):

    import re
    def add_extra_newline(yamltext):
        # newlines not followed by a space
        pattern = r'\n(?! )'
        # extra newline
        modified_text = re.sub(pattern, '\n\n', yamltext)
        return modified_text

    with open(filename, 'w') as f:
        yamltext = add_extra_newline(nested_dict_yamlout(d, index, **kwargs))
        f.write(yamltext)
    if print_text:
        print(yamltext)

    return yamltext

# define global dicts
rel_var = 1e-2
param_variations = {
        'PRC/length_PRM_PR2': {'var': 0.005, 'offset': 0, 'type': 'abs'},
        'PRC/length_PR2_PR3': {'var': 0.005, 'offset': 0, 'type': 'abs'},
        'PRC/length_PR3_BS': {'var': 0.005, 'offset': 0, 'type': 'abs'},
        'PRC/PR2/Rc': {'var': rel_var, 'offset': 0, 'type': 'rel'},
        'PRC/PR3/Rc': {'var': rel_var, 'offset': 0, 'type': 'rel'},
        'PRC/PRM/Rc': {'var': rel_var, 'offset': 0, 'type': 'rel'},
        # values below this are direct entries and not additive variations
        'beam_waist/w0x': {'var': 0.0005, 'offset': 0.001, 'type': 'abs'},
        'beam_waist/zx': {'var': 0.5, 'offset': 6, 'type': 'abs'},
        'beam_waist/w0y': {'var': 0.0005, 'offset': 0.001, 'type': 'abs'},
        'beam_waist/zy': {'var': 0.5, 'offset': 6, 'type': 'abs'},
        'loss/carm': {'var': 2.5, 'offset': 77.5, 'type': 'abs'},
        'loss/darm': {'var': 2.5, 'offset': 2.5, 'type': 'abs'},
        'loss/PRCL': {'var': 200, 'offset': 300, 'type': 'abs'},
        'absorption/ITMX': {'var': 0.25e-6, 'offset': 0.35e-6, 'type': 'abs'},
        'absorption/ITMY': {'var': 0.25e-6, 'offset': 0.35e-6, 'type': 'abs'},
        'absorption/ETMX': {'var': 0.25e-6, 'offset': 0.35e-6, 'type': 'abs'},
        'absorption/ETMY': {'var': 0.25e-6, 'offset': 0.35e-6, 'type': 'abs'},
        'RH_eff/ITMX': {'var': 1.25, 'offset': 1.75, 'type': 'abs'},
        'RH_eff/ITMY': {'var': 1.25, 'offset': 1.75, 'type': 'abs'},
        'RH_eff/ETMX': {'var': 0.2, 'offset': 0.7, 'type': 'abs'},
        'RH_eff/ETMY': {'var': 0.2, 'offset': 0.7, 'type': 'abs'},
        }

# extra model parameters to record
model_output_params = ['PRM.p1.o', 'PR2.p2.o', 'PR3.p2.o',
                       'BS.p1.i', 'BS.p2.i', 'BSARAS.p3.i', 'BSARAS.p4.i',
                       'ITMXAR.p1.i', 'ITMX.p1.o', 'ETMX.p1.i', 'ITMY.p1.o', 'ETMY.p1.i',
                       'SR3.p1.i', 'SR2.p1.i', 'SRM.p1.o', 'gouy', 'gouy_xarm', 'gouy_yarm']

# Trapeziodal reward function defined as follows (on either side of centre) -
#         plateu
#         v
# _________       <- weight
#          \
#           \
#            \
#             \
#              \_____ <- 0
#               \
#                \
#                 \
#                  \
# ^            ^
# centre       base

reward_params = {'PRG': {'timestamps': [40, 1500, 5000],
                        'weight': [1, 1, 1],
                        'centre': [37, 0.86, 0.92],
                        'plateu': [1.85, 0.01, 0.01],
                        'base': [9.25, 0.03, 0.03],
                        'type': ['abs', 'rel', 'rel'],
                        },
                'PRG_noRH': {'timestamps': [40, 1500, 5000],
                        'weight': [0, 0, 0],
                        'centre': [37, 37, 37],
                        'plateu': [1.85, 1.85, 1.85],
                        'base': [9.25, 9.25, 9.25],
                        'type': ['abs', 'abs', 'abs'],
                        },
                'PRG9': {'timestamps': [40, 1500, 5000],
                        'weight': [1, 1, 1],
                        'centre': [85, 1.235, 1.106],
                        'plateu': [4.25, 0.01, 0.01],
                        'base': [21.25, 0.03, 0.03],
                        'type': ['abs', 'rel', 'rel'],
                        },
                'PRG9_noRH': {'timestamps': [40, 1500, 5000],
                        'weight': [1, 0, 0],
                        'centre': [115, 115, 115],
                        'plateu': [5.75, 5.75, 5.75],
                        'base': [28.75, 28.75, 28.75],
                        'type': ['abs', 'abs', 'abs'],
                        },
                'PreflPRM': {'timestamps': [40, 1500, 5000],
                        'weight': [0, 0, 0],
                        'centre': [0.047, 0.047, 0.047],
                        'plateu': [0.00235, 0.00235, 0.00235],
                        'base': [0.01175, 0.01175, 0.01175],
                        'type': ['abs', 'abs', 'abs'],
                        },
                'PreflPRM_noRH': {'timestamps': [40, 1500, 5000],
                        'weight': [0, 0, 0],
                        'centre': [0.047, 0.047, 0.047],
                        'plateu': [0.00235, 0.00235, 0.00235],
                        'base': [0.01175, 0.01175, 0.01175],
                        'type': ['abs', 'abs', 'abs'],
                        },
                'gouy': {'timestamps': [40, 1500, 5000],
                        'weight': [1, 0, 0],
                        'centre': [21, 21, 21],
                        'plateu': [1.05, 1.05, 1.05],
                        'base': [5.25, 5.25, 5.25],
                        'type': ['abs', 'abs', 'abs'],
                        },
                'gouy_noRH': {'timestamps': [40, 1500, 5000],
                        'weight': [0, 0, 0],
                        'centre': [21, 21, 21],
                        'plateu': [1.05, 1.05, 1.05],
                        'base': [5.25, 5.25, 5.25],
                        'type': ['abs', 'abs', 'abs'],
                        },
                }

