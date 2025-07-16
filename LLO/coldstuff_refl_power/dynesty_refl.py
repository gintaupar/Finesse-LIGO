# %%
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
import corner
import numpy as np
import sys
import os

import mpi4py
mpi4py.rc.threads = False
mpi4py.rc.recv_mprobe = False

from schwimmbad import MPIPool
import dynesty
from dynesty.utils import resample_equal
from numbers import Number
import finesse
from scipy.stats import norm, uniform
import numpy as np
import matplotlib.pyplot as plt
import finesse
from finesse.utilities.maps import circular_aperture
import finesse_ligo
from finesse_ligo.factory import aligo
import finesse.analysis.actions as fa
from finesse_ligo.maps import get_test_mass_surface_profile_interpolated
from finesse_ligo.actions import InitialLockLIGO
import numpy as np
from finesse.knm import Map
import finesse
from finesse.ligo import git_path
import matplotlib.pyplot as plt
import gpstime
import warnings
from finesse.analysis.actions.base import AnalysisState

import dill as pickle
pickle.settings['recurse'] = True

from types import SimpleNamespace

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

finesse.init_plotting()

import sys
import os

prefix = '/fred/oz147/'

if is_notebook():
    checkpoint = None
else:
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-c", "--checkpoint", default=None, dest="checkpoint", help="checkpoint file to use")
    parser.add_option("-d", "--dlogz", default=0.1, dest="dlogz", help="dlogz to target")
    
    (options, args) = parser.parse_args()
    
    checkpoint = options.checkpoint    
    dlogz = options.dlogz    

# %%
ifo = 'LLO'
factory = aligo.ALIGOFactory(git_path() / ifo / "yaml" / f"{ifo.lower()}_O4.yaml")
factory.options.INPUT.add_IMC_and_IM1 = False
model = factory.make()

model.PR3.R = 1-model.PR3.L.ref
model.PR3.T = 0
model.PR3.L = 5.3e-6

# Change default gauss to be at PRM input, propagate IMC more to PRM AR input
qx_prm = model.propagate_beam(model.L0.p1.o, model.PRMAR.p2.i, direction='x').q(model.PRMAR.p2.i)
qy_prm = model.propagate_beam(model.L0.p1.o, model.PRMAR.p2.i, direction='y').q(model.PRMAR.p2.i)

QS = finesse.BeamParam.overlap_contour(qx_prm, 0.1, np.linspace(0, 2*np.pi))
W0 = np.array([q.w0 for q in QS])
Z = np.array([q.z for q in QS])
W0_min_mm = W0.min()/1e-3
W0_max_mm = W0.max()/1e-3
Z_min     = Z.min()
Z_max     = Z.max()

model.remove(model.gIM2_p1_i)
model.add(
    finesse.components.Gauss(
        'g1',
        model.PRMAR.p2.i,
        w0=qx_prm.w0,
        z=qx_prm.z,
    )
)

RH_ITM_Rc_uD_W = 1e-6 # uD/W ITM
RH_ETM_Rc_uD_W = 0.84e-6 # uD/W ETM
RH_ITM_lens_uD_W = -9.9e-6 # uD/W ITM

initial_ITMX_Rc = float(model.ITMX.Rcx)
initial_ETMX_Rc = float(model.ETMX.Rcx)
initial_ITMY_Rc = float(model.ITMY.Rcx)
initial_ETMY_Rc = float(model.ETMY.Rcx)
initial_X_lens = float(model.ITMXlens.f)
initial_Y_lens = float(model.ITMYlens.f)

def set_thermal_state(P_IX_RH, P_IY_RH, P_EX_RH, P_EY_RH):
    model.ITMX.Rc = 2 / (2 / abs(initial_ITMX_Rc) + RH_ITM_Rc_uD_W * P_IX_RH)
    model.ETMX.Rc = 2 / (2 / abs(initial_ETMX_Rc) + RH_ETM_Rc_uD_W * P_EX_RH)
    model.ITMY.Rc = 2 / (2 / abs(initial_ITMY_Rc) + RH_ITM_Rc_uD_W * P_IY_RH)
    model.ETMY.Rc = 2 / (2 / abs(initial_ETMY_Rc) + RH_ETM_Rc_uD_W * P_EY_RH)

    model.ITMXlens.f = 1/(1/initial_X_lens + RH_ITM_lens_uD_W * P_IX_RH)
    model.ITMYlens.f = 1/(1/initial_Y_lens + RH_ITM_lens_uD_W * P_IY_RH)

set_thermal_state(0.7*1, 0.7*1, 0.7*2.2, 0.7*2.2)

model.L0.P = 2
R = 0.17
N = 301

x, y = (
    np.linspace(-R, R, N),
    np.linspace(-R, R, N),
)
mask = finesse.utilities.maps.circular_aperture(x, y, R)

# Get surfaces
for TM in [model.ITMX, model.ITMY, model.ETMX, model.ETMY]:
    if 'X' in TM.name:
        P = factory.params.X
    else:
        P = factory.params.Y
    TM._unfreeze()
    TM.static = get_test_mass_surface_profile_interpolated(P[TM.name[:-1]].ID, make_axisymmetric=True)(x, y)
    TM._freeze()

model.ITMX.surface_map = Map(x, y, amplitude=mask, opd=model.ITMX.static)
model.ETMX.surface_map = Map(x, y, amplitude=mask, opd=model.ETMX.static)
model.ITMY.surface_map = Map(x, y, amplitude=mask, opd=model.ITMY.static)
model.ETMY.surface_map = Map(x, y, amplitude=mask, opd=model.ETMY.static)

model.ITMXlens.OPD_map = Map(x, y, amplitude=mask)
model.ITMYlens.OPD_map = Map(x, y, amplitude=mask)

x_r3, XR3_baffle = finesse.ligo.maps.aligo_O4_PR3_SR3_baffle(N=N)

#model.PR3.surface_map = Map(x_r3, x_r3, amplitude=XR3_baffle)
#model.SR3.surface_map = Map(x_r3, x_r3, amplitude=XR3_baffle)
model.PR3.surface_map = Map(x_r3, x_r3, amplitude=circular_aperture(x_r3, x_r3, 0.13))
model.SR3.surface_map = Map(x_r3, x_r3, amplitude=circular_aperture(x_r3, x_r3, 0.13))
# ESD aperture
model.ITMXAR.surface_map = Map(x_r3, x_r3, amplitude=circular_aperture(x_r3, x_r3, 0.13))
model.ITMYAR.surface_map = Map(x_r3, x_r3, amplitude=circular_aperture(x_r3, x_r3, 0.13))

# compute the round trip losses with the maps in and make sure overall loss is reasonable
model.modes("even", maxtem=4)
model.parse("fd E_car_refl PRM.p2.o f=0")
model.parse("fd E_u9_refl PRM.p2.o f=+f1")
model.parse("fd E_l9_refl PRM.p2.o f=-f1")
model.parse("fd E_car_prc PRM.p1.o f=0")
model.parse("fd E_u9_prc PRM.p1.o f=+f1")
model.parse("fd E_l9_prc PRM.p1.o f=-f1")
eigx = model.run("eigenmodes(cavXARM, 0)")
eigy = model.run("eigenmodes(cavYARM, 0)")

model.X_arm_loss = 70e-6
model.Y_arm_loss = 70e-6

loss_x = (model.X_arm_loss + eigx.loss(True)[1][0])
loss_y = (model.Y_arm_loss + eigy.loss(True)[1][0])

#print("Adjusting arm losses to match requested values")
#print("X arm loss: ", loss_x/1e-6, "ppm")
#print("Y arm loss: ", loss_y/1e-6, "ppm")
## Apply corrections to get back to original losses
#print("Old X arm plane-wave loss: ", model.X_arm_loss/1e-6, "ppm")
#print("Old Y arm plane-wave loss: ", model.Y_arm_loss/1e-6, "ppm")
model.X_arm_loss -= eigx.loss(True)[1][0]
model.Y_arm_loss -= eigy.loss(True)[1][0]
#print("New X arm plane-wave loss: ", model.X_arm_loss/1e-6, "ppm")
#print("New Y arm plane-wave loss: ", model.Y_arm_loss/1e-6, "ppm")
model.phase_level = 2 # phase 2 so HG00 gouy phase is removed from length phasing which makes finding error signal easier

lock = InitialLockLIGO(
    exception_on_lock_fail=True,
    lock_steps=100,
    gain_scale=0.5,
    pseudo_lock_arms=False,
    run_locks=True,
    exception_on_check_fail=True,
)

sol = model.run(fa.Series(lock, fa.Noxaxis(name="DC")))

model.g1.priority = 1e10
model.cavPRX.priority = 1e10-1
model.cavPRY.priority = 1e10-2
model.parse("mathd PRG_full Pprc/Pin")

model.CARM_lock.accuracy *= 10
model.PRCL_lock.accuracy *= 10
model.SRCL_lock.accuracy *= 10
model.DARM_rf_lock.accuracy *= 10
model.MICH_lock.accuracy *= 10

base = model.deepcopy()
state = None
# %%
# Actions to run on the model for each sample
run_locks = fa.RunLocks(max_iterations=200)
noxaxis = fa.Noxaxis()
actions = fa.Series(run_locks, noxaxis)
# Get the changing parameters
memo = {"changing_parameters":[
    'ITMX.Rcx', 'ITMX.Rcy', 'ITMY.Rcx', 'ITMY.Rcy',
    'ETMX.Rcx', 'ETMX.Rcy', 'ETMY.Rcx', 'ETMY.Rcy',
    'ITMXlens.f', 'ITMYlens.f']}

run_locks._requests(model, memo)
#lock._requests(model, memo)
memo['changing_parameters'] = np.unique(memo['changing_parameters']).tolist()

# %%
def map_params(params):
    obj = SimpleNamespace()
    obj.PRM = params[0]
    obj.PR2 = params[1]
    obj.PR3 = params[2]
    obj.dl12 = params[3]
    obj.dl23 = params[4]
    obj.carm_loss = params[5]
    obj.darm_loss = params[6]
    obj.w0 = params[7]
    obj.z = params[8]
    obj.RHX_eff = params[9]
    obj.RHY_eff = params[10]
    #obj.extra_PRC_loss = params[11]
    return obj

def run_model(params, initial_lock=False):
    global model
    global state
    
    for p in memo['changing_parameters']:
        model.get(p).value = base.get(p).value
        
    obj = map_params(params)
    
    model.PRM.Rc = obj.PRM
    model.PR2.Rc = obj.PR2
    model.PR3.Rc = obj.PR3
    model.lp1.L = base.lp1.L + obj.dl12 * 1e-3
    model.lp2.L = base.lp2.L + obj.dl23 * 1e-3
    model.lp3.L = base.lp3.L - obj.dl12 * 1e-3 - obj.dl23 * 1e-3
    model.X_arm_loss = (obj.carm_loss + obj.darm_loss) * 1e-6
    model.Y_arm_loss = (obj.carm_loss - obj.darm_loss) * 1e-6
    #model.PR3.L = obj.extra_PRC_loss * 1e-6
    
    model.g1.w0x = obj.w0 * 1e-3
    model.g1.w0y = obj.w0 * 1e-3
    model.g1.zx = obj.z
    model.g1.zy = obj.z
    
    set_thermal_state(
        obj.RHX_eff*1,
        obj.RHY_eff*1,
        obj.RHX_eff*2.2,
        obj.RHY_eff*2.2,
    )
    
    # # Solve all detectors
    if initial_lock:
        state.apply(lock)
    else:
        #try:
        state.apply(run_locks)
        # except finesse.exceptions.LostLock:
        #     #print("initial")
        #     for p in memo['changing_parameters']:
        #         model.get(p).value = base.get(p).value
        #     # Try again with initial lock
        #     state.apply(lock)
            
    sol = state.apply(noxaxis)
    return sol
    
def prior_transform(u):
    x = np.zeros_like(u)
    # Try measured uncertainties from Gari, uncertainty in RoC is around 1mm
    x[0] = norm.ppf(u[0], loc=-11.009, scale=15e-3) # PRM https://dcc.ligo.org/LIGO-T1300740
    x[1] = norm.ppf(u[1], loc=-4.548, scale=4.2e-3) # PR2 https://dcc.ligo.org/LIGO-T1200116
    x[2] = norm.ppf(u[2], loc=+36.027, scale=1e-3) # PR3 https://dcc.ligo.org/LIGO-E1101065
    
    x[3] = 5 * (2 * u[3] - 1) # delta PR1-2 distance [mm]
    x[4] = 10 * (2 * u[4] - 1) # delta PR2-3 distance [mm]

    x[5] = (40 * u[5] + 50) # carm_loss [ppm]
    x[6] = 10*(2 * u[6] - 1) # darm_loss [ppm]
    # 20% mismatch range
    x[7] = ((W0_max_mm - W0_min_mm) * u[7] + W0_min_mm) # w0 [mm]
    x[8] = ((Z_max - Z_min) * u[8] + Z_min) # z [m]
    x[9] = uniform.ppf(u[9], loc=0.0, scale=1) # RHX eff [%]
    x[10] = uniform.ppf(u[10], loc=0.0, scale=1) # RHY eff [%]
    
    #x[11] = uniform.ppf(u[11], loc=0, scale=100) # extra PRC loss [ppm]
    return x

LIKELIHOODS = {
    'PRG': (norm, {'loc':37, 'scale': 0.5}),
    'gouy': (norm, {'loc':21, 'scale': 1}),
    'Prefl': (norm, {'loc':55, 'scale': 2}),
    'PRG9': (norm, {'loc':95, 'scale': 2}),
}

def loglikli(**parameters): 
    # Get an error if we don't assert that it's a float
    # ValueError: setting an array element with a sequence. The requested array
    # has an inhomogeneous shape after 1 dimensions. The detected shape was
    # (20,) + inhomogeneous part.
    return np.sum(
        [LIKELIHOODS[k][0].logpdf(float(v), **LIKELIHOODS[k][1]) for k,v in parameters.items()]
    )
    
def lnLikelihood(params):
    blob = np.zeros(1, dtype=[
        ('PRG', float),
        ('gouy', float),
        ('Prefl', float),
        ('Pin', float),
        ('Px', float),
        ('Py', float),
        ('gouy_xarm', float),
        ('gouy_yarm', float),
        ('PRG9', float),
        ('PRG45', float),
        ('PRX_length', float),
        ('PRY_length', float),
        ('input_prc_mismatch_x', float),
        ('input_prc_mismatch_y', float),
        ('itm_lens_uD_x', float),
        ('itm_lens_uD_y', float),
    ])
    
    def run(initial_lock=False):
        sol = run_model(params, initial_lock=initial_lock)
        blob['gouy'] = np.mean([model.cavPRX.gouy/2, model.cavPRY.gouy/2]) # roundtrip/2 for single pass
        blob['PRG'] = sol["PRG_full"]
        blob['PRG9'] = sol["PRG9"]
        blob['PRG45'] = sol["PRG45"]
        blob['Prefl'] = sol["PreflPRM"]/1e-3 # mW
        blob['Pin'] = sol["Pin"]
        blob['Px'] = sol['Px']
        blob['Py'] = sol['Py']
        blob['gouy_xarm'] = np.mean(model.cavXARM.gouy)/2 # roundtrip/2 for single pass
        blob['gouy_yarm'] = np.mean(model.cavYARM.gouy)/2 # roundtrip/2 for single pass
        blob['PRX_length'] = float(model.cavPRX.path.physical_length)
        blob['PRY_length'] = float(model.cavPRY.path.physical_length)
        blob['itm_lens_uD_x'] = (1/float(model.ITMXlens.f))/1e-6
        blob['itm_lens_uD_y'] = (1/float(model.ITMYlens.f))/1e-6
        
        q1x = model.propagate_beam(model.g1.node, model.PRMAR.p2.o, model.PRM.p2.i, direction='x').q(model.PRMAR.p2.o)
        q1y = model.propagate_beam(model.g1.node, model.PRMAR.p2.o, model.PRM.p2.i, direction='y').q(model.PRMAR.p2.o)
        q2x = model.propagate_beam(model.PRM.p1.i, model.PRMAR.p2.o, direction='x').q(model.PRMAR.p2.o)
        q2y = model.propagate_beam(model.PRM.p1.i, model.PRMAR.p2.o, direction='y').q(model.PRMAR.p2.o)
        
        blob['input_prc_mismatch_x'] = finesse.BeamParam.mismatch(q1x, q2x)
        blob['input_prc_mismatch_y'] = finesse.BeamParam.mismatch(q1y, q2y)
        
        L = loglikli(
            PRG=blob['PRG'],
            gouy=blob['gouy'],
            Prefl=blob['Prefl'],
            PRG9=blob['PRG9'],
        )
        #print(params)  
        #print(model.X_arm_loss/1e-6, model.Y_arm_loss/1e-6)
        #print(float(blob['PRG']), float(blob['gouy']), float(blob['Prefl']), float(blob['PRG9']), L)  
        return L

    try:
        return run(False), blob    
    except (finesse_ligo.exceptions.InitialLockCheckException, finesse.warnings.CavityUnstableWarning):
        return -np.inf, blob # Too much geometry changed
    except finesse.exceptions.LostLock:
        #print("lost lock")
        try:
            return -np.inf, blob # run(True) # Try an initial lock, if that fails it doesn't work
        except (finesse.exceptions.LostLock, finesse_ligo.exceptions.InitialLockCheckException):
            return -np.inf, blob

# %%
warnings.simplefilter("error", category=finesse.warnings.CavityUnstableWarning)

labels=[
    'PRM Rc [m]',
    'PR2 Rc [m]',
    'PR3 Rc [m]',
    'dL12 [mm]',
    'dL23 [mm]',
    'carm_loss [ppm]',
    'darm_loss [ppm]',
    'w0 [mm]',
    'z [m]',
    'RHX scale [frac]',
    'RHY scale [frac]',
    #'Extra PRC loss [ppm]'
]

time = os.environ.get('SLURM_JOB_ID', 0)
    
# Make a new analysis state (Simulation) that we can apply actions on
# to generate solutions. This saves rebuilding the model each sample
state = AnalysisState(model)
state.build_model(
    [
        model.PRM.Rcx,
        model.PR2.Rcx,
        model.PR3.Rcx,
        model.PRM.Rcy,
        model.PR2.Rcy,
        model.PR3.Rcy,
        model.lp1.L,
        model.lp2.L,
        model.lp3.L,
        model.X_arm_loss,
        model.Y_arm_loss,
        model.g1.w0x,
        model.g1.zx,
        model.g1.w0y,
        model.g1.zy,
        model.PR3.L,
        *(model.get(_) for _ in memo["changing_parameters"]),
    ],
    []
)

try:
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    # if True:
    #     pool = None
        print("SLURM_JOB_ID ", time, " started at ", gpstime.gpsnow())
        if checkpoint is None:
            nlive = 4000    # number of live points
            bound = 'multi'   # use MutliNest algorithm for bounds
            ndims = len(labels)
            #bootstrap = 0
            assert ndims == len(labels)
            
            sample = 'rwalk' # random walk sampling
            update_interval = 4.2
            
            sampler = dynesty.NestedSampler(
                lnLikelihood,
                prior_transform,
                ndims,
                bound=bound,
                sample=sample,
                nlive=nlive,
                pool=pool,
                update_interval=update_interval,
                #bootstrap=bootstrap,
                blob=True,
            )
            sampler.run_nested(checkpoint_file=f'{prefix}dynesty.{rank}.{time}.save', print_progress=True, dlogz=dlogz)
        else:
            checkpoint_file = f'{prefix}dynesty.0.{checkpoint}.save'
            
            print("Using checkpoint time", checkpoint)
            print("Using checkpoint file", checkpoint_file)
            try:
                sampler = dynesty.NestedSampler.restore(
                    checkpoint_file,
                    pool=pool
                )
                sampler.run_nested(
                    checkpoint_file=f'{prefix}dynesty.{rank}.{time}.save',
                    print_progress=True,
                    resume=True,
                    dlogz=dlogz
                )
            except UserWarning:
                pass # ignore warnings and just run, like if we restart a finsihed run
        
        print("done")
        res = sampler.results  # get results dictionary from sampler
        
        # draw posterior samples
        weights = np.exp(res['logwt'] - res['logz'][-1])
        posterior_samples = resample_equal(res.samples, weights)
        np.savez_compressed(
            f"{prefix}dynesty_{rank}_{time}_samples.npz",
            posterior_samples=posterior_samples,
            weights=weights,
            labels=labels,
            parameters={
                p.full_name:
                    p.value.eval() if hasattr(p.value,'eval') else (p.value if isinstance(p.value, Number) else None)
                for p in model.all_parameters
            },
            blob=res['blob'],
        )
        print("save", f"{prefix}dynesty_{rank}_{time}_samples.npz")
        
        # Save prior for future reference
        with open(f"{prefix}dynesty_{rank}_{time}_prior_transform.pkl", 'wb') as f:
            pickle.dump(prior_transform, f)
        with open(f"{prefix}dynesty_{rank}_{time}_likelihood.pkl", 'wb') as f:
            pickle.dump(loglikli, f)
        with open(f"{prefix}dynesty_{rank}_{time}_LIKELIHOODS.pkl", 'wb') as f:
            pickle.dump(LIKELIHOODS, f)
            
        print('Number of posterior samples is {}'.format(len(posterior_samples)))

        corner.corner(
            posterior_samples,
            labels=labels, show_titles=True, color='k'
        )
        
        import pathlib
        pathlib.Path(f"{prefix}figures/{time}").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{prefix}figures/{time}/corner.pdf")
        print("save", f"{prefix}figures/{time}/corner.pdf")
finally:
    state.finished()

