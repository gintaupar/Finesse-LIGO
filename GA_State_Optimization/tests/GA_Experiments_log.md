### Experiments
1. A general run 
    - `N = 200, generations = 30, topX = 0.3, abs_var = 5e-3, rel_var = 1e-2`
    - `add_apertures = True, frac_crossover = 0.5, frac_mutation = 0.8, frac_shrink = 0.3`
    - results: `genetic_results_Jun23.pkl`
    - (tentative) reward_params = `{'AGX': {'weight': 1, 'centre': 300, 'plateu': 20, 'base': 30},
                 'AGY': {'weight': 1, 'centre': 300, 'plateu': 20, 'base': 30},
                 'Px': {'weight': 1, 'centre': 10000, 'plateu': 2000, 'base': 3000},
                 'Py': {'weight': 1, 'centre': 10000, 'plateu': 2000, 'base': 3000},
                 'PRG': {'weight': 1, 'centre': 40, 'plateu': 3, 'base': 10},
                 'PRG9': {'weight': 1, 'centre': 115, 'plateu': 15, 'base': 40},
                 }`
    - `AGX/Y` were certainly wrong here, as in reality it is pretty confidently known to be 265. The recovered solutions however landed on this value automatically.
    - The solutions show very low Prefl values (<1mW) as compared to reality (~50mW).
    - The rise in PRG9 from 103 to 116 (for thermalised state of mirrors) while going from RH to no RH is something desirable. Howevery, the solutions here are obtained for cold state with RH at 90% efficiency. In the final solutions, PRG9 varies between 104-115 with RHs compared to 84 in real detector in cold state. With RHs off, it rises by ~4-5 cts, a bit lower than the extent of expected rise.

2. Including Prefl constraint and removing Px Py (to remove dependence on input power)
    - changes: `generations = 15, topX = 0.4, frac_shrink = 0.3`
    - results: `genetic_results_27Jun_REWARDS-AGXY_PRG_PRG9_Prefl.pkl`
    - reward_params = `{'AGX': {'weight': 1, 'centre': 260, 'plateu': 5, 'base': 10},
                 'AGY': {'weight': 1, 'centre': 260, 'plateu': 5, 'base': 10},
                 'PRG': {'weight': 1, 'centre': 38, 'plateu': 4, 'base': 6},
                 'PRG9': {'weight': 1, 'centre': 100, 'plateu': 20, 'base': 40},
                 'Prefl': {'weight': 1, 'centre': 0.05, 'plateu': 0.025, 'base': 0.035},
                 }`

3. Forcing RH to no RH in cold state with tentative PRG9 for no RH around 105 along with decent ~50mW Prefl (small run)
    - changes from prev: `N = 200, generations = 15, topX = 0.3, frac_shrink = 0.5`
    - results: `genetic_results_27Jun_REWARDS-AGXY_PRG_PRG9_PRG9noRH_Prefl.pkl`
    - reward_params = `{'AGX': {'weight': 0.25, 'centre': 260, 'plateu': 5, 'base': 10},
                 'AGY': {'weight': 0.25, 'centre': 260, 'plateu': 5, 'base': 10},
                 'PRG': {'weight': 1, 'centre': 38, 'plateu': 1, 'base': 3},
                 'PRG9': {'weight': 1, 'centre': 90, 'plateu': 5, 'base': 10},
                 'PRG9_noRH': {'weight': 1, 'centre': 105, 'plateu': 5, 'base': 10},
                 'Prefl': {'weight': 1, 'centre': 0.05, 'plateu': 0.025, 'base': 0.035},
                 }`

4. Forcing REFL power to match (small run)
    - changes from prev: `N = 20, generations = 10, topX = 0.3, frac_shrink = 0.5`
    - results: `genetic_results_28Jun_REWARDS-AGXY_PRG_Prefl.pkl`
    - reward_params = `{'AGX': {'weight': 0.33, 'centre': 260, 'plateu': 5, 'base': 10},
                 'AGY': {'weight': 0.33, 'centre': 260, 'plateu': 5, 'base': 10},
                 'PRG': {'weight': 0.33, 'centre': 38, 'plateu': 5, 'base': 10},
                 'Prefl': {'weight': 2, 'centre': 0.05, 'plateu': 0.015, 'base': 0.025},
                 }`

5. Including waist params of gauss beam and carm, darm losses in the input params and gouy phase in the output params (gouy nan replaced by 0)
    N = 200
    w0var = 0.1
    zvar = 1e-3
    lvar = 8 # in ppm
    - param_variations = `{
                'INPUT/length_IM4_PRM_AR': {'min': -abs_var, 'max': abs_var, 'offset': 0, 'type': 'abs'},
                'PRC/length_PRM_PR2': {'min': -abs_var, 'max': abs_var, 'offset': 0, 'type': 'abs'},
                'PRC/length_PR2_PR3': {'min': -abs_var, 'max': abs_var, 'offset': 0, 'type': 'abs'},
                'PRC/length_PR3_BS': {'min': -abs_var, 'max': abs_var, 'offset': 0, 'type': 'abs'},
                'beam_waist/w0x': {'min': -w0var, 'max': w0var, 'offset': 0, 'type': 'rel'},
                'beam_waist/w0y': {'min': -w0var, 'max': w0var, 'offset': 0, 'type': 'rel'},
                'beam_waist/zx': {'min': -zvar, 'max': zvar, 'offset': 0, 'type': 'abs'},
                'beam_waist/zy': {'min': -zvar, 'max': zvar, 'offset': 0, 'type': 'abs'},
                'PRC/PR2/Rc': {'min': -rel_var, 'max': rel_var, 'offset': 0, 'type': 'rel'},
                'PRC/PR3/Rc': {'min': -rel_var, 'max': rel_var, 'offset': 0, 'type': 'rel'},
                'PRC/PRM/Rc': {'min': -rel_var, 'max': rel_var, 'offset': 0, 'type': 'rel'},
                'loss/carm': {'min': -lvar, 'max': lvar, 'offset': 65, 'type': 'abs'},
                'loss/darm': {'min': -lvar, 'max': lvar, 'offset': 0, 'type': 'abs'},
                }`
    - reward_params = `{'PRG': {'weight': 1, 'centre': 38, 'plateu': 5, 'base': 10},
                'PRG9': {'weight': 1, 'centre': 90, 'plateu': 5, 'base': 10},
                'Prefl': {'weight': 1, 'centre': 0.05, 'plateu': 0.015, 'base': 0.025},
                'gouy': {'weight': 1, 'centre': 21, 'plateu': 2, 'base': 10},
                }`
    - model_output_params = `['PRM.p1.o', 'PR2.p2.o', 'PR3.p2.o', 'BS.p1.i', 'BS.p2.i', 'BSARAS.p3.i', 'BSARAS.p4.i',
                        'ITMXAR.p1.i', 'ITMX.p1.o', 'ETMX.p1.i', 'ITMY.p1.o', 'ETMY.p1.i',
                        'SR3.p1.i', 'SR2.p1.i', 'SRM.p1.o', 'gouy', 'gouy_xarm', 'gouy_yarm']`
    - results: `genetic_results200_31Jun_REWARDS-AGXY_PRG_PRG9_Prefl_gouy.pkl`

6. Somewhat realistic ranges; waist and waist position vars are on the scale of those obtained from 1% change in RoC at PRM, also defined wrt PRMAR now (w0=1.0175e-3, z=4.48); arm losses to be kept in range 70-85ppm; to be run on fully parallelised script on cluster
    N = 10000
    - param_variations = `{
                'INPUT/length_IM4_PRM_AR': {'min': -0.005, 'max': 0.005, 'offset': 0, 'type': 'abs'},
                'PRC/length_PRM_PR2': {'min': -0.005, 'max': 0.005, 'offset': 0, 'type': 'abs'},
                'PRC/length_PR2_PR3': {'min': -0.005, 'max': 0.005, 'offset': 0, 'type': 'abs'},
                'PRC/length_PR3_BS': {'min': -0.005, 'max': 0.005, 'offset': 0, 'type': 'abs'},
                'beam_waist/w0x': {'min': -0.008, 'max': 0.008, 'offset': 0, 'type': 'abs'},
                'beam_waist/w0y': {'min': -0.008, 'max': 0.008, 'offset': 0, 'type': 'abs'},
                'beam_waist/zx': {'min': -0.1, 'max': 0.1, 'offset': 0, 'type': 'abs'},
                'beam_waist/zy': {'min': -0.1, 'max': 0.1, 'offset': 0, 'type': 'abs'},
                'PRC/PR2/Rc': {'min': -rel_var, 'max': rel_var, 'offset': 0, 'type': 'rel'},
                'PRC/PR3/Rc': {'min': -rel_var, 'max': rel_var, 'offset': 0, 'type': 'rel'},
                'PRC/PRM/Rc': {'min': -rel_var, 'max': rel_var, 'offset': 0, 'type': 'rel'},
                'loss/carm': {'min': 75, 'max': 80, 'offset': 0, 'type': 'abs'},
                'loss/darm': {'min': -5, 'max': 5, 'offset': 0, 'type': 'abs'},
                }`
    - reward_params = `{'PRG': {'weight': 1, 'centre': 38, 'plateu': 5, 'base': 10},
                'PRG9': {'weight': 1, 'centre': 90, 'plateu': 5, 'base': 10},
                'Prefl': {'weight': 1, 'centre': 0.05, 'plateu': 0.015, 'base': 0.025},
                'gouy': {'weight': 1, 'centre': 21, 'plateu': 2, 'base': 10},
                }`
    - model_output_params = `['PRM.p1.o', 'PR2.p2.o', 'PR3.p2.o', 'BS.p1.i', 'BS.p2.i', 'BSARAS.p3.i', 'BSARAS.p4.i',
                        'ITMXAR.p1.i', 'ITMX.p1.o', 'ETMX.p1.i', 'ITMY.p1.o', 'ETMY.p1.i',
                        'SR3.p1.i', 'SR2.p1.i', 'SRM.p1.o', 'gouy', 'gouy_xarm', 'gouy_yarm']`
    - results: `run_20240722` on CIT

7. Run: run_20240722b; changed reward for 'Prefl': {'weight': 1, 'centre': 0.05, 'plateu': 0.01, 'base': 0.05}, increased number of gens to 15.

8. Run: run_20240725: same as above but without BS_ITM and BS_HR apertures

9. Run: 240729: 
    ### Note Added Later: RH_control was wrong! This should be the efficiency ie 0.9 (still the states obtained gave descent results)

    MAXTEM = 12
    Beam params updated as per Anamaria's suggestion: w0=1.015e-3, z=6.0
    RH_control = {'ITMX': 1.2, 'ITMY': 1.2, 'ETMX': 2.2, 'ETMY': 2.2}
    reward_params = {'PRG': {'weight': 1, 'centre': 38, 'plateu': 1, 'base': 5},
                    'PRG9': {'weight': 1, 'centre': 90, 'plateu': 5, 'base': 10},
                    'Prefl': {'weight': 1, 'centre': 0.05, 'plateu': 0.01, 'base': 0.05},
                    'gouy': {'weight': 1, 'centre': 21, 'plateu': 2, 'base': 10},
                    }
    --total_generations 15 --ids_per_simulation 20 --population_size 5000 --job_file_dir ${repopath}/job_files/jobs_${suffix} --repopath ${repopath} --suffix ${suffix} --top_x_percent 0.05 --frac_crossover 0.5 --frac_mutation 0.8 --frac_shrink 0.5 --add_TM_apertures 1 --add_PR3_SR3_aperture 1 --add_BS_ITM_aperture 1 --add_BS_HR_aperture 1

10. 240729b: RH_control was wrong in prev run! RH_efficiency set to 0.9

11. 240730: manually setting w0=1.0175e-3, z=4.48 just to see the effect. Results look bad. So, new beam params are crucial.

12. 240730b: Using the better beam params again (w0=1.015e-3, z=6.0); starting a large run. 
    MAXTEM = 8
    # including 'PRG45' monitor as well
    reward_params = {'PRG': {'weight': 1, 'centre': 37, 'plateu': 1, 'base': 8},
                    'PRG9': {'weight': 1, 'centre': 85, 'plateu': 5, 'base': 15},
                    'PRG45': {'weight': 1, 'centre': 11, 'plateu': 1, 'base': 5},
                    'PreflPRM': {'weight': 1, 'centre': 0.047, 'plateu': 0.01, 'base': 0.025},
                    'gouy': {'weight': 1, 'centre': 21, 'plateu': 2, 'base': 10},
                    }
    # introducing free PRCL loss param in param_variations: 'loss/PRCL': {'min': 100, 'max': 200, 'offset': 0, 'type': 'abs'}

13. 240731: w0 was earlier varying as 'rel' and not 'abs'. So only small fractions. Now made to vary in abs full range. PRC loss range increased to 100-500ppm.

14. 240801: params like w0y, zy, length_IM4_PRM_AR were redundant as the Gaussian beam is not defined as astigmatic and it is defined wrt PRMAR. Also, PRC loss was not going into the model correctly earlier - fixed now.

15. 240802: reintroduced astigmatism (correctly this time) to keep the effects of angular incidences on the mirrors to be accounted for. Slightly tighter reward for PRG9: 'PRG9': {'weight': 1, 'centre': 85, 'plateu': 3, 'base': 15}

16. 240807: including params after 5k secs of thermal evolution
    reward_params = {'PRG': {'weight': 1, 'centre': 37, 'plateu': 1, 'base': 8},
                'PRG9': {'weight': 1, 'centre': 85, 'plateu': 3, 'base': 15},
                'PRG45': {'weight': 1, 'centre': 11, 'plateu': 1, 'base': 5},
                'PreflPRM': {'weight': 1, 'centre': 0.047, 'plateu': 0.01, 'base': 0.025},
                'gouy': {'weight': 1, 'centre': 21, 'plateu': 2, 'base': 10},
                'PreflPRM_thermalised': {'weight': 1, 'centre': 2.5, 'plateu': 0.3, 'base': 3},
                'PRG_thermalised': {'weight': 1, 'centre': 35, 'plateu': 3, 'base': 8},
                'PRG9_thermalised': {'weight': 1, 'centre': 95, 'plateu': 5, 'base': 10},
                'PRG45_thermalised': {'weight': 1, 'centre': 10, 'plateu': 1, 'base': 5},
                }
    - Pulled aggresive reward func at gen8. So should see different evolution after that. Abandoning run at gen16 as things had deviated away quite a bit already.

17. 240809: aggresive reward func for avoiding suboptimal solutions that compromise a few params way too much. Also including absorptivities of TMs as free params.
    param_variations = {
            'PRC/length_PRM_PR2': {'min': -0.005, 'max': 0.005, 'offset': 0, 'type': 'abs'},
            'PRC/length_PR2_PR3': {'min': -0.005, 'max': 0.005, 'offset': 0, 'type': 'abs'},
            'PRC/length_PR3_BS': {'min': -0.005, 'max': 0.005, 'offset': 0, 'type': 'abs'},
            'beam_waist/w0x': {'min': -0.0005, 'max': 0.005, 'offset': 0, 'type': 'abs'},
            'beam_waist/zx': {'min': -0.2, 'max': 0.2, 'offset': 0, 'type': 'abs'},
            'beam_waist/w0y': {'min': -0.0005, 'max': 0.005, 'offset': 0, 'type': 'abs'},
            'beam_waist/zy': {'min': -0.2, 'max': 0.2, 'offset': 0, 'type': 'abs'},
            'PRC/PR2/Rc': {'min': -rel_var, 'max': rel_var, 'offset': 0, 'type': 'rel'},
            'PRC/PR3/Rc': {'min': -rel_var, 'max': rel_var, 'offset': 0, 'type': 'rel'},
            'PRC/PRM/Rc': {'min': -rel_var, 'max': rel_var, 'offset': 0, 'type': 'rel'},
            'loss/carm': {'min': 75, 'max': 80, 'offset': 0, 'type': 'abs'},
            'loss/darm': {'min': -5, 'max': 5, 'offset': 0, 'type': 'abs'},
            'loss/PRCL': {'min': 100, 'max': 500, 'offset': 0, 'type': 'abs'},
            'absorption/ITMX': {'min': 0.1e-6, 'max': 0.6e-6, 'offset': 0, 'type': 'abs'},
            'absorption/ITMY': {'min': 0.1e-6, 'max': 0.6e-6, 'offset': 0, 'type': 'abs'},
            'absorption/ETMX': {'min': 0.1e-6, 'max': 0.6e-6, 'offset': 0, 'type': 'abs'},
            'absorption/ETMY': {'min': 0.1e-6, 'max': 0.6e-6, 'offset': 0, 'type': 'abs'},
            'RH_eff/ITMX': {'min': 0.5, 'max': 3., 'offset': 0, 'type': 'abs'},
            'RH_eff/ITMY': {'min': 0.5, 'max': 3., 'offset': 0, 'type': 'abs'},
            'RH_eff/ETMX': {'min': 0.5, 'max': 0.9, 'offset': 0, 'type': 'abs'},
            'RH_eff/ETMY': {'min': 0.5, 'max': 0.9, 'offset': 0, 'type': 'abs'},
            }
    - Shifted to LLO
    - ./src/generate_dag.py --total_generations 20 --ids_per_simulation 5 --population_size 2500 --job_file_dir ${repopath}/job_files/jobs_${suffix} --repopath ${repopath} --suffix ${suffix} --top_x_percent 0.1 --frac_crossover 0.5 --frac_mutation 0.8 --frac_shrink 0.5 --add_TM_apertures 1 --add_PR3_SR3_aperture 1 --add_BS_ITM_aperture 1 --add_BS_HR_aperture 1

18. 240809b: No RH variation included to isolate just abs effects. So it is default 0.9 for all TMs.
    - ./src/generate_dag.py --total_generations 20 --ids_per_simulation 20 --population_size 10000 --job_file_dir ${repopath}/job_files/jobs_${suffix} --repopath ${repopath} --suffix ${suffix} --top_x_percent 0.05 --frac_crossover 0.5 --frac_mutation 0.8 --frac_shrink 0.5 --add_TM_apertures 1 --add_PR3_SR3_aperture 1 --add_BS_ITM_aperture 1 --add_BS_HR_aperture 1

19. 240810: Cold reward to be matched first - if normed reward for cold > 0.8 only then thermalisation is run; tighter reward func 
    - reward_params = {'PRG': {'weight': 1, 'centre': 37, 'plateu': 1, 'base': 8},
                'PRG9': {'weight': 1, 'centre': 85, 'plateu': 3, 'base': 15},
                'PRG45': {'weight': 1, 'centre': 11, 'plateu': 1, 'base': 5},
                'PreflPRM': {'weight': 1, 'centre': 0.047, 'plateu': 0.01, 'base': 0.025},
                'gouy': {'weight': 1, 'centre': 21, 'plateu': 2, 'base': 10},
                'PreflPRM_thermalised': {'weight': 1, 'centre': 2.5, 'plateu': 0.3, 'base': 3},
                'PRG_thermalised': {'weight': 1, 'centre': 34, 'plateu': 2, 'base': 8},
                'PRG9_thermalised': {'weight': 1, 'centre': 92, 'plateu': 2, 'base': 10},
                'PRG45_thermalised': {'weight': 1, 'centre': 10, 'plateu': 1, 'base': 5},
                }

20. 240811: Super narrow param variation range inspired from 240802 solutions, only allowing absorption and RH eff to vary majorly now;
    - param_variations = {
        'PRC/length_PRM_PR2': {'min': -0.006796, 'max': -0.004096, 'offset': 0, 'type': 'abs'},
        'PRC/length_PR2_PR3': {'min': -0.003472, 'max': -0.002095, 'offset': 0, 'type': 'abs'},
        'PRC/length_PR3_BS': {'min': -0.005448, 'max': -0.003725, 'offset': 0, 'type': 'abs'},
        'beam_waist/w0x': {'min': -0.000151, 'max': 0.000105, 'offset': 0, 'type': 'abs'},
        'beam_waist/zx': {'min': -0.189255, 'max': 0.206195, 'offset': 0, 'type': 'abs'},
        'beam_waist/w0y': {'min': -0.000195, 'max': -2.7e-05, 'offset': 0, 'type': 'abs'},
        'beam_waist/zy': {'min': -0.136258, 'max': 0.145566, 'offset': 0, 'type': 'abs'},
        'PRC/PR2/Rc': {'min': -0.000229, 'max': 0.000523, 'offset': 0, 'type': 'abs'},
        'PRC/PR3/Rc': {'min': -0.000529, 'max': -0.000435, 'offset': 0, 'type': 'abs'},
        'PRC/PRM/Rc': {'min': -0.004588, 'max': 0.007099, 'offset': 0, 'type': 'abs'},
        'loss/carm': {'min': 75.189099, 'max': 77.800012, 'offset': 0, 'type': 'abs'},
        'loss/darm': {'min': -5.264082, 'max': 6.918534, 'offset': 0, 'type': 'abs'},
        'loss/PRCL': {'min': 590.24028, 'max': 673.276965, 'offset': 0, 'type': 'abs'},
        'absorption/ITMX': {'min': 0.1e-6, 'max': 0.6e-6, 'offset': 0, 'type': 'abs'},
        'absorption/ITMY': {'min': 0.1e-6, 'max': 0.6e-6, 'offset': 0, 'type': 'abs'},
        'absorption/ETMX': {'min': 0.1e-6, 'max': 0.6e-6, 'offset': 0, 'type': 'abs'},
        'absorption/ETMY': {'min': 0.1e-6, 'max': 0.6e-6, 'offset': 0, 'type': 'abs'},
        'RH_eff/ITMX': {'min': 0.5, 'max': 3., 'offset': 0, 'type': 'abs'},
        'RH_eff/ITMY': {'min': 0.5, 'max': 3., 'offset': 0, 'type': 'abs'},
        'RH_eff/ETMX': {'min': 0.5, 'max': 0.9, 'offset': 0, 'type': 'abs'},
        'RH_eff/ETMY': {'min': 0.5, 'max': 0.9, 'offset': 0, 'type': 'abs'},
        }
    - ./src/generate_dag.py --total_generations 20 --ids_per_simulation 5 --population_size 10000 --job_file_dir ${repopath}/job_files/jobs_${suffix} --repopath ${repopath} --suffix ${suffix} --top_x_percent 0.05 --frac_crossover 0.5 --frac_mutation 0.8 --frac_shrink 0.5 --add_TM_apertures 1 --add_PR3_SR3_aperture 1 --add_BS_ITM_aperture 1 --add_BS_HR_aperture 1

21. 240826: In all the older runs the RH effects were getting overriden by the thermalisation. The bug is corrected in this run. Going back to wider param variations - 
    - param_variations = {
            'PRC/length_PRM_PR2': {'min': -0.005, 'max': 0.005, 'offset': 0, 'type': 'abs'},
            'PRC/length_PR2_PR3': {'min': -0.005, 'max': 0.005, 'offset': 0, 'type': 'abs'},
            'PRC/length_PR3_BS': {'min': -0.005, 'max': 0.005, 'offset': 0, 'type': 'abs'},
            'beam_waist/w0x': {'min': -0.0005, 'max': 0.005, 'offset': 0, 'type': 'abs'},
            'beam_waist/zx': {'min': -0.2, 'max': 0.2, 'offset': 0, 'type': 'abs'},
            'beam_waist/w0y': {'min': -0.0005, 'max': 0.005, 'offset': 0, 'type': 'abs'},
            'beam_waist/zy': {'min': -0.2, 'max': 0.2, 'offset': 0, 'type': 'abs'},
            'PRC/PR2/Rc': {'min': -rel_var, 'max': rel_var, 'offset': 0, 'type': 'rel'},
            'PRC/PR3/Rc': {'min': -rel_var, 'max': rel_var, 'offset': 0, 'type': 'rel'},
            'PRC/PRM/Rc': {'min': -rel_var, 'max': rel_var, 'offset': 0, 'type': 'rel'},
            'loss/carm': {'min': 75, 'max': 80, 'offset': 0, 'type': 'abs'},
            'loss/darm': {'min': -5, 'max': 5, 'offset': 0, 'type': 'abs'},
            'loss/PRCL': {'min': 100, 'max': 500, 'offset': 0, 'type': 'abs'},
            'absorption/ITMX': {'min': 0.1e-6, 'max': 0.6e-6, 'offset': 0, 'type': 'abs'},
            'absorption/ITMY': {'min': 0.1e-6, 'max': 0.6e-6, 'offset': 0, 'type': 'abs'},
            'absorption/ETMX': {'min': 0.1e-6, 'max': 0.6e-6, 'offset': 0, 'type': 'abs'},
            'absorption/ETMY': {'min': 0.1e-6, 'max': 0.6e-6, 'offset': 0, 'type': 'abs'},
            'RH_eff/ITMX': {'min': 0.5, 'max': 3., 'offset': 0, 'type': 'abs'},
            'RH_eff/ITMY': {'min': 0.5, 'max': 3., 'offset': 0, 'type': 'abs'},
            'RH_eff/ETMX': {'min': 0.5, 'max': 0.9, 'offset': 0, 'type': 'abs'},
            'RH_eff/ETMY': {'min': 0.5, 'max': 0.9, 'offset': 0, 'type': 'abs'},
            }
    - frac crossover changed to 0.3 as mixing of solutions was largely giving low rewards in prev runs.
        - ./src/generate_dag.py --total_generations 20 --ids_per_simulation 5 --population_size 10000 --job_file_dir ${repopath}/job_files/jobs_${suffix} --repopath ${repopath} --suffix ${suffix} --top_x_percent 0.05 --frac_crossover 0.3 --frac_mutation 0.8 --frac_shrink 0.5 --add_TM_apertures 1 --add_PR3_SR3_aperture 1 --add_BS_ITM_aperture 1 --add_BS_HR_aperture 1

22. 240909: Trying to capture the PRG9 bump by recording additional data points at t_rec=1500
    - reward_params = {'PRG': {'weight': 1, 'centre': 37, 'plateu': 1, 'base': 8},
                'PRG9': {'weight': 1, 'centre': 85, 'plateu': 3, 'base': 15},
                'PRG45': {'weight': 1, 'centre': 11, 'plateu': 1, 'base': 5},
                'PreflPRM': {'weight': 1, 'centre': 0.047, 'plateu': 0.01, 'base': 0.025},
                'gouy': {'weight': 1, 'centre': 21, 'plateu': 2, 'base': 10},
                'PreflPRM_thermalised': {'weight': 1, 'centre': 2.5, 'plateu': 0.3, 'base': 3},
                'PRG_thermalised': {'weight': 1, 'centre': 34, 'plateu': 2, 'base': 8},
                'PRG9_thermalised': {'weight': 1, 'centre': 92, 'plateu': 2, 'base': 10},
                'PRG45_thermalised': {'weight': 1, 'centre': 10, 'plateu': 1, 'base': 5},
                'PreflPRM_rec': {'weight': 1, 'centre': 2.5, 'plateu': 1, 'base': 3},
                'PRG_rec': {'weight': 1, 'centre': 32, 'plateu': 2, 'base': 8},
                'PRG9_rec': {'weight': 1, 'centre': 105, 'plateu': 5, 'base': 10},
                'PRG45_rec': {'weight': 1, 'centre': 10, 'plateu': 2, 'base': 5},
                }
    - --frac_crossover increased to 0.5

23. 241007: Including new BS given by Cao; Keeping everything else the same for now; all old style aperture options removed as we are implementing them anyway.

24. 241104: cold reward thresh reduced to 0 from 0.8. Rerunning.

25. 241113: reward function adjusted for slope that is 20% relative to the centre. Also, omitting 45MHz sideband data.
    - reward_params = {'PRG': {'weight': 1, 'centre': 37, 'plateu': 1, 'base': 8.4},
                'PRG9': {'weight': 1, 'centre': 85, 'plateu': 3, 'base': 20},
                'PreflPRM': {'weight': 1, 'centre': 0.047, 'plateu': 0.01, 'base': 0.0194},
                'gouy': {'weight': 1, 'centre': 21, 'plateu': 2, 'base': 6.2},
                'PreflPRM_thermalised': {'weight': 1, 'centre': 2.5, 'plateu': 0.3, 'base': 0.8},
                'PRG_thermalised': {'weight': 1, 'centre': 34, 'plateu': 2, 'base': 8.8},
                'PRG9_thermalised': {'weight': 1, 'centre': 92, 'plateu': 2, 'base': 20.4},
                'PreflPRM_rec': {'weight': 1, 'centre': 2.5, 'plateu': 1, 'base': 1.5},
                'PRG_rec': {'weight': 1, 'centre': 32, 'plateu': 2, 'base': 8.4},
                'PRG9_rec': {'weight': 1, 'centre': 105, 'plateu': 5, 'base': 26},
                }

26. 241118: reward function abs in first time step and relative in later times. For ['abs', 'rel', 'rel'] - 
    - PRG: [37, 32, 34] -> [37, 0.86, 0.92]
    - PRG9: [85, 105, 94] -> [85, 1.235, 1.106]
    - reward_params =  {'PRG': {'timestamps': [250, 1500, 5000],
                                'weight': [1, 1, 1],
                                'centre': [37, 0.86, 0.92],
                                'plateu': [1, 0.01, 0.01],
                                'base': [8.4, 0.03, 0.03],
                                'type': ['abs', 'rel', 'rel']},
                        'PRG9': {'timestamps': [250, 1500, 5000],
                                'weight': [1, 1, 1],
                                'centre': [85, 1.235, 1.106],
                                'plateu': [3, 0.01, 0.01],
                                'base': [20, 0.03, 0.03],
                                'type': ['abs', 'rel', 'rel']},
                        }

27. 241127: changed weightages for PRG and PRG9 for getting correct DRMI state.
    - 'weight': [3, 1, 1]

28. 250126: running DRMI tests with RH vs no_RH; main change is increased PRG9
    - reward_params = {'PRG': {'timestamps': [250],
                        'weight': [1],
                        'centre': [37],
                        'plateu': [1],
                        'base': [8.4],
                        'type': ['abs']},
                'PRG9': {'timestamps': [250],
                        'weight': [2],
                        'centre': [85],
                        'plateu': [3],
                        'base': [20],
                        'type': ['abs']},
                'PreflPRM': {'timestamps': [250],
                        'weight': [1],
                        'centre': [0.047],
                        'plateu': [0.01],
                        'base': [0.0194],
                        'type': ['abs']},
                'PRG9_noRH': {'timestamps': [250],
                        'weight': [2],
                        'centre': [115],
                        'plateu': [5],
                        'base': [28],
                        'type': ['abs']},
                }
    - midway at gen 12, made the following changes
        - 'timestamps': [40]
        - all 'weight': [1]
        - reduced thermal_evol_time to 60

29. 20250129: Balanced reward_params. Only tracking PRG, PRG9 and PRG9_noRH for DRMI state. Added more observables with 0 weight for easier visualization. Faster runs with timestamp 40 and thermal run up to 60.
    - reward_params = {'PRG': {'timestamps': [40],
                        'weight': [1],
                        'centre': [37],
                        'plateu': [1.85],
                        'base': [9.25],
                        'type': ['abs']},
                'PRG_noRH': {'timestamps': [40],
                        'weight': [0],
                        'centre': [37],
                        'plateu': [1.85],
                        'base': [9.25],
                        'type': ['abs']},
                'PRG9': {'timestamps': [40],
                        'weight': [1],
                        'centre': [85],
                        'plateu': [4.25],
                        'base': [21.25],
                        'type': ['abs']},
                'PRG9_noRH': {'timestamps': [40],
                        'weight': [1],
                        'centre': [115],
                        'plateu': [5.75],
                        'base': [28.75],
                        'type': ['abs']},
                'PreflPRM': {'timestamps': [40],
                        'weight': [0],
                        'centre': [0.047],
                        'plateu': [0.00235],
                        'base': [0.01175],
                        'type': ['abs']},
                'PreflPRM_noRH': {'timestamps': [40],
                        'weight': [0],
                        'centre': [0.047],
                        'plateu': [0.00235],
                        'base': [0.01175],
                        'type': ['abs']},
                'gouy': {'timestamps': [40],
                        'weight': [0],
                        'centre': [21],
                        'plateu': [1.05],
                        'base': [5.25],
                        'type': ['abs']},
                'gouy_noRH': {'timestamps': [40],
                        'weight': [1],
                        'centre': [21],
                        'plateu': [1.05],
                        'base': [5.25],
                        'type': ['abs']},
                }

29. 20250203: waist param range corrected: (-0.0005, 0.0005). Small debug in ITM lens update_maps. Slower shrink, lower crossover, higher top_frac.
    - gouy_noRH weight: 1
    - ./src/generate_dag.py --total_generations 20 --ids_per_simulation 20 --population_size 10000 --job_file_dir ${repopath}/job_files/jobs_${suffix} --repopath ${repopath} --suffix ${suffix} --top_x_percent 0.1 --frac_crossover 0.3 --frac_mutation 0.8 --frac_shrink 0.7

30. 20250206: simplified and debugged crossover (consistent parent params); reward function for DRMI + Thermalization
    - reward_params = {'PRG': {'timestamps': [40, 1500, 5000],
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

31. 20250209: corrected ranges in param variations. No negative values allowed for waist params, losses and absorptions (but allowed for RH effs). Waist params also go in as direct entries, just like losses, abs, RH eff etc.
        - paramd variations between -var to +var (fixes shrink factor) instead of min and max.
        - param_variations = {
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

32. 20250213: Independent additional ITM lenses. Externally controlled run evol times. Cold state RH vs no_RH run. 10 iters per job.
        - --thermal_evol_time 60 --cold_evol_time 60 --ids_per_simulation 10
        - new additions in param_variations:
                - 'P_lens_addl/ITMX': {'var': 40e-6, 'offset': 0., 'type': 'abs'}
                - 'P_lens_addl/ITMY': {'var': 40e-6, 'offset': 0., 'type': 'abs'}
        - reward_params = 'PRG': 'weight': [1, 0, 0]
                          'PRG_noRH': 'weight': [1, 0, 0]
                          'PRG9': 'weight': [1, 0, 0]
                          'PRG9_noRH': 'weight': [1, 0, 0]
                          'PreflPRM': 'weight': [1, 0, 0]
                          'PreflPRM_noRH': 'weight': [1, 0, 0]
                          'gouy': 'weight': [1, 0, 0]
                          'gouy_noRH': 'weight': [0, 0, 0]

33. 20250214: new full run for indep ITM lenses. 
        - reward_params = 'PRG': 'weight': [2, 1, 1]
                        'PRG_noRH': 'weight': [2, 0, 0]
                        'PRG9': 'weight': [2, 1, 1]
                        'PRG9_noRH': 'weight': [2, 0, 0]
                        'PreflPRM': 'weight': [0.5, 0, 0]
                        'PreflPRM_noRH': 'weight': [0.5, 0, 0]
                        'gouy': 'weight': [0.5, 0, 0]

34. 20250224: addl const surface deformation; starting around the cold state of 20250213; small run of 1000 points/gen
        - changes in param_variations : 
                'RH_eff/ITMX': {'var': 0.05, 'offset': 0.95, 'type': 'abs'},
                'RH_eff/ITMY': {'var': 0.05, 'offset': 0.95, 'type': 'abs'},
                'RH_eff/ETMX': {'var': 0.2, 'offset': 0.7, 'type': 'abs'},
                'RH_eff/ETMY': {'var': 0.2, 'offset': 0.7, 'type': 'abs'},
                'P_lens_addl/ITMX': {'var': 10e-6, 'offset': -13e-6, 'type': 'abs'},
                'P_lens_addl/ITMY': {'var': 10e-6, 'offset': -13e-6, 'type': 'abs'},
                'P_surfdef_addl/ITMX': {'var': 5e-6, 'offset': 0., 'type': 'abs'},
                'P_surfdef_addl/ITMY': {'var': 5e-6, 'offset': 0., 'type': 'abs'},
                'P_surfdef_addl/ETMX': {'var': 5e-6, 'offset': 0., 'type': 'abs'},
                'P_surfdef_addl/ETMY': {'var': 5e-6, 'offset': 0., 'type': 'abs'},
        - changes in reward_params : 
                'PRG': 'weight': [2, 0, 0]
                'PRG_noRH': 'weight': [2, 0, 0]
                'PRG9': 'weight': [2, 0, 0]
                'PRG9_noRH': 'weight': [2, 0, 0]
                'PreflPRM': 'weight': [1, 0, 0]
                'PreflPRM_noRH': 'weight': [1, 0, 0]
                'gouy': 'weight': [1, 0, 0]
                'gouy_noRH': 'weight': [0, 0, 0]
