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
import finesse
from finesse_ligo.factory import aligo
import numpy as np
import finesse
from finesse.ligo import git_path
import matplotlib.pyplot as plt
import warnings
import gpstime

import dill as pickle
pickle.settings['recurse'] = True


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import sys
import os

if is_notebook():
    checkpoint = None
    dlogz = 1
    prefix = '/fred/oz147/LLO_single_bounce/'
else:
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-c", "--checkpoint", default=None, dest="checkpoint", help="checkpoint file to use")
    parser.add_option("-d", "--dlogz", default=1, dest="dlogz", help="dlogz to target")
    parser.add_option("-p", "--prefix", dest="prefix", help="where to store")
    parser.add_option("-n", "--nlive", dest="nlive", help="Live points")
    
    (options, args) = parser.parse_args()
    
    prefix = options.prefix
    checkpoint = options.checkpoint    
    dlogz = float(options.dlogz)
    nlive = int(options.nlive)
    
    if not prefix.endswith("/"):
        prefix += "/"
        
    if not os.path.exists(prefix):
        os.mkdir(prefix)

# %%
ifo = 'LLO'
factory = aligo.ALIGOFactory(git_path() / ifo / "yaml" / f"{ifo.lower()}_O4.yaml")
factory.options.INPUT.add_IMC_and_IM1 = True

model = factory.make()

model.remove('cavXARM')
model.remove('cavYARM')
model.remove('cavPRX')
model.remove('cavPRY')
model.remove('cavSRX')
model.remove('cavSRY')

initial_ITMX_Rc = float(model.ITMX.Rcx)
initial_ETMX_Rc = float(model.ETMX.Rcx)
initial_ITMY_Rc = float(model.ITMY.Rcx)
initial_ETMY_Rc = float(model.ETMY.Rcx)
initial_X_lens = float(model.ITMXlens.f)
initial_Y_lens = float(model.ITMYlens.f)

# RH_ITM_Rc_uD_W = 1e-6 # uD/W ITM
# RH_ETM_Rc_uD_W = 0.84e-6 # uD/W ETM
# RH_ITM_lens_uD_W = -9.9e-6 # uD/W ITM

# Actual with cage numbers - around about 20% higher
RH_ITM_Rc_uD_W = 1.2e-6 # uD/W ITM
RH_ITM_lens_uD_W = -12e-6 # uD/W ITM

model.add_parameter("RH_eff", 0.8)
model.add_parameter("w0", 1e-3)
model.add_parameter("z", -0.28)

model.add_parameter("P_IX_RH", 0)
model.add_parameter("P_IY_RH", 0)
model.add_parameter("P_EX_RH", 0)
model.add_parameter("P_EY_RH", 0)

model.ITMX.Rc = 2 / (2 / abs(initial_ITMX_Rc) + RH_ITM_Rc_uD_W * model.P_IX_RH.ref * model.RH_eff.ref)
#model.ETMX.Rc = 2 / (2 / abs(initial_ETMX_Rc) + RH_ETM_Rc_uD_W * model.P_EX_RH.ref * model.RH_eff.ref)
model.ITMY.Rc = 2 / (2 / abs(initial_ITMY_Rc) + RH_ITM_Rc_uD_W * model.P_IY_RH.ref * model.RH_eff.ref)
#model.ETMY.Rc = 2 / (2 / abs(initial_ETMY_Rc) + RH_ETM_Rc_uD_W * model.P_EY_RH.ref * model.RH_eff.ref)

model.ITMXlens.f = 1/(1/initial_X_lens + RH_ITM_lens_uD_W * model.P_IX_RH.ref * model.RH_eff.ref)
model.ITMYlens.f = 1/(1/initial_Y_lens + RH_ITM_lens_uD_W * model.P_IY_RH.ref * model.RH_eff.ref)
    
base = model.deepcopy()

propx = model.propagate_beam(
    from_node=model.MC3.p3.o,
    to_node=model.AS.p1.i,
    via_node=model.ITMX.p1.i,
    direction='x',
)

focal_lengths = [c for c in propx.components if hasattr(c, 'f')]
# Ignore first component where we start
curvatures = [c for c in propx.components[1:] if hasattr(c, 'Rcx') and c.Rcx != np.inf]

for comp in curvatures:
    comp.Rcy = comp.Rcx.ref
    
# %%
from itertools import pairwise
from functools import reduce

def make_q_propagator(from_node, to_node, via_node, direction):
    nr1 = float(from_node.space.nr) if from_node.space is not None else 1
    nr2 = float(to_node.space.nr) if to_node.space is not None else 1

    path = model.path(
        from_node=from_node,
        to_node=to_node,
        via_node=via_node,
    )

    ABCDs = np.zeros((len(path.components), 2, 2), dtype=float)

    def run():
        for i, ((a, b), comp) in enumerate(zip(pairwise(path.nodes), path.components)):
            ABCDs[i] = comp.ABCD(a, b, direction=direction)
            
        ABCD = reduce(np.matmul, ABCDs[::-1])
        
        q_in = finesse.BeamParam(w0=model.w0, z=model.z)
            
        return q_in.transform(ABCD, nr1=nr1, nr2=nr2)
    
    return run

from_node = model.PRMAR.p1.o
to_node = model.AS.p1.i
    
q_IMC_IX_DCPD_x = make_q_propagator(from_node, to_node, model.ITMX.p1.i, 'x')
q_IMC_IX_DCPD_y = make_q_propagator(from_node, to_node, model.ITMX.p1.i, 'y')
q_IMC_IY_DCPD_x = make_q_propagator(from_node, to_node, model.ITMY.p1.i, 'x')
q_IMC_IY_DCPD_y = make_q_propagator(from_node, to_node, model.ITMY.p1.i, 'y')
    
def get_w0_z_range(q):
    qs = finesse.BeamParam.overlap_contour(
        q,
        0.08, np.linspace(0, 2*np.pi, 1000)
    )

    W0 = np.vectorize(lambda q: q.w0)(qs)
    Z = np.vectorize(lambda q: q.z)(qs)

    return W0.min(), W0.max(), Z.min(), Z.max()

W0_min, W0_max, Z_min, Z_max = get_w0_z_range(from_node.qx)

# %%
def mismatch_IX():
    overlap_x = finesse.BeamParam.overlap(
        q_IMC_IX_DCPD_x(),
        to_node.qx
    )
    overlap_y = finesse.BeamParam.overlap(
        q_IMC_IX_DCPD_y(),
        to_node.qy
    )
    return 1 - np.sqrt(overlap_x * overlap_y)

def mismatch_IY():
    overlap_x = finesse.BeamParam.overlap(
        q_IMC_IY_DCPD_x(),
        to_node.qx
    )
    overlap_y = finesse.BeamParam.overlap(
        q_IMC_IY_DCPD_y(),
        to_node.qy
    )
    return 1 - np.sqrt(overlap_x * overlap_y)

# %%
initial_values = {
    "PRM.Rcx": factory.params.PRC.PRM.Rc,
    "PR2.Rcx": factory.params.PRC.PR2.Rc,
    "PR3.Rcx": factory.params.PRC.PR3.Rc,
    "SRM.Rcx": factory.params.SRC.SRM.Rc, 
    "SR2.Rcx": factory.params.SRC.SR2.Rc, 
    "SR3.Rcx": factory.params.SRC.SR3.Rc,
    "OM1.Rcx": factory.params.OUTPUT.OM1.Rc,
    "OM2.Rcx": factory.params.OUTPUT.OM2.Rc,
    #"IM2.Rcx": factory.params.INPUT.IM2.Rc,
    #"IM3.Rcx": factory.params.INPUT.IM3.Rc,
    "lp1.L"  : factory.params.PRC.length_PRM_PR2, 
    "lp2.L"  : factory.params.PRC.length_PR2_PR3,
    "ls1.L"  : factory.params.SRC.length_SRM_SR2, 
    "ls2.L"  : factory.params.SRC.length_SR2_SR3,
    "OFI_p3__OM1_p1.L": factory.params.OUTPUT.length_OFI_OM1,
    "OM1_p2__OM2_p1.L": factory.params.OUTPUT.length_OM1_OM2,
    "OM1_p2__OM2_p1.L": factory.params.OUTPUT.length_OM1_OM2,
    "OM3_p2__OMC_IC_p1.L": factory.params.OUTPUT.length_OM3_OMC,
    #"IM2_p2__IFI_p1.L": factory.params.INPUT.length_IM2_IFI,
}

priors = {
    "RH_eff": lambda u: uniform.ppf(u, loc=0.7, scale=0.2),
    "w0": lambda u: uniform.ppf(u, loc=W0_min, scale=W0_max-W0_min),
    "z": lambda u: uniform.ppf(u, loc=Z_min, scale=Z_max-Z_min),
    "PRM.Rcx": lambda u: norm.ppf(u, loc=initial_values["PRM.Rcx"], scale=15e-3), # PRM https://dcc.ligo.org/LIGO-T1300740
    "PR2.Rcx": lambda u: norm.ppf(u, loc=initial_values["PR2.Rcx"], scale=4.2e-3), # PR2 https://dcc.ligo.org/LIGO-T1200116
    "PR3.Rcx": lambda u: norm.ppf(u, loc=initial_values["PR3.Rcx"], scale=1e-3), # PR3 https://dcc.ligo.org/LIGO-E1101065
    # Guess SR uncertainties are similar to PR
    "SRM.Rcx": lambda u: norm.ppf(u, loc=initial_values["SRM.Rcx"], scale=2e-3), 
    "SR2.Rcx": lambda u: norm.ppf(u, loc=initial_values["SR2.Rcx"], scale=3e-3), 
    "SR3.Rcx": lambda u: norm.ppf(u, loc=initial_values["SR3.Rcx"], scale=11e-3), # https://dcc.ligo.org/LIGO-E1101231
    
    #"IM2.Rcx": lambda u: norm.ppf(u, loc=initial_values["IM2.Rcx"], scale=5e-3),
    #"IM3.Rcx": lambda u: norm.ppf(u, loc=initial_values["IM3.Rcx"], scale=5e-3),
    "OM1.Rcx": lambda u: norm.ppf(u, loc=initial_values["OM1.Rcx"], scale=10e-3),
    "OM2.Rcx": lambda u: norm.ppf(u, loc=initial_values["OM2.Rcx"], scale=10e-3),
    
    # lengths all off by some mm?
    "lp1.L": lambda u: initial_values["lp1.L"] + uniform.ppf(u, loc=-1, scale=2) * 5e-3, 
    "lp2.L": lambda u: initial_values["lp2.L"] + uniform.ppf(u, loc=-1, scale=2) * 5e-3,
    "ls1.L": lambda u: initial_values["ls1.L"] + uniform.ppf(u, loc=-1, scale=2) * 5e-3, 
    "ls2.L": lambda u: initial_values["ls2.L"] + uniform.ppf(u, loc=-1, scale=2) * 5e-3,
    
    "OFI_p3__OM1_p1.L": lambda u: initial_values["OFI_p3__OM1_p1.L"] + uniform.ppf(u, loc=-1, scale=2) * 10e-3,
    "OM1_p2__OM2_p1.L": lambda u: initial_values["OM1_p2__OM2_p1.L"] + uniform.ppf(u, loc=-1, scale=2) * 10e-3,
    "OM3_p2__OMC_IC_p1.L": lambda u: initial_values["OM3_p2__OMC_IC_p1.L"] + uniform.ppf(u, loc=-1, scale=2) * 10e-3,
    #"IM2_p2__IFI_p1.L": lambda u: norm.ppf(u, loc=initial_values["IM2_p2__IFI_p1.L"], scale=10e-3),
}

RH = {
    "OFF": 0,
    "ON": 0.78
}

# mismatch_measurements = {
#     "XOFF": (norm, {'loc':0.1175, 'scale': 0.005}),
#     "XON": (norm, {'loc':0.16, 'scale': 0.001}),
#     "YOFF": (norm, {'loc':(0.122+0.121)/2, 'scale': abs(0.122-0.121)}),
#     "YON": (norm, {'loc':(0.156+0.153)/2, 'scale': abs(0.156-0.153)}),
# }
mismatch_measurements = {
    "mismatch_RHX_on": (norm, {'loc':(0.161+0.159)/2, 'scale': 0.005}),
    "mismatch_RHX_off": (norm, {'loc':(0.118+0.117)/2, 'scale': 0.005}),
    "mismatch_RHY_on": (norm, {'loc':(0.156+0.153)/2, 'scale': 0.005}),
    "mismatch_RHY_off": (norm, {'loc':(0.127+0.127)/2, 'scale': 0.005}),
}

# %%
blob = np.zeros(1, dtype=[
    ('mismatch_RHX_on', float),
    ('mismatch_RHY_on', float),
    ('mismatch_RHX_off', float),
    ('mismatch_RHY_off', float),
])

def prior_transform(u):
    x = np.zeros_like(u)
    for i, prior in enumerate(priors.values()):
        x[i] = prior(u[i])
    return x

def loglikli(**parameters):
    return np.sum(
        [mismatch_measurements[k][0].logpdf(float(v), **mismatch_measurements[k][1]) for k, v in parameters.items()]
    )
    
def lnLikelihood(params):
    try:
        for i, parameter in enumerate(priors):
            model.get(parameter).value = params[i]
        
        model.P_IX_RH = RH['ON']
        model.P_IY_RH = RH['ON']
        blob['mismatch_RHX_on'] = mismatch_RHX_on = mismatch_IX()
        blob['mismatch_RHY_on'] = mismatch_RHY_on = mismatch_IY()
        
        model.P_IX_RH = RH['OFF']
        model.P_IY_RH = RH['OFF']
        blob['mismatch_RHX_off'] = mismatch_RHX_off = mismatch_IX()
        blob['mismatch_RHY_off'] = mismatch_RHY_off = mismatch_IY()
        
        L = loglikli(
            mismatch_RHX_off=mismatch_RHX_off,
            mismatch_RHX_on=mismatch_RHX_on,
            mismatch_RHY_off=mismatch_RHY_off,
            mismatch_RHY_on=mismatch_RHY_on,
        )
        #print(params, L)
        return L, blob
    except finesse.warnings.CavityUnstableWarning:
        return -np.inf, blob
    
# %%
warnings.simplefilter("error", category=finesse.warnings.CavityUnstableWarning)

bound = 'multi'   # use MutliNest algorithm for bounds
ndims = len(priors)

sample = 'rslice' # random walk sampling
update_interval = 4.2

jobid = os.environ.get('SLURM_JOB_ID', int(gpstime.gpsnow()))
labels = tuple(priors.keys())

with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    sampler = dynesty.NestedSampler(
        lnLikelihood,
        prior_transform,
        ndims,
        bound=bound,
        sample=sample,
        nlive=nlive,
        update_interval=update_interval,
        pool=pool,
        blob=True,
    )
    sampler.run_nested(print_progress=True, dlogz=dlogz)
    print()
    print("Done")
    print()
    res = sampler.results  # get results dictionary from sampler
    
    # draw posterior samples
    weights = np.exp(res['logwt'] - res['logz'][-1])
    posterior_samples = resample_equal(res.samples, weights)
    np.savez_compressed(
        f"{prefix}dynesty_{rank}_{jobid}_samples.npz",
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
    print("save", f"{prefix}dynesty_{jobid}_samples.npz")
    
    # Save prior for future reference
    with open(f"{prefix}dynesty_{jobid}_prior_transform.pkl", 'wb') as f:
        pickle.dump(prior_transform, f)
    with open(f"{prefix}dynesty_{jobid}_likelihood.pkl", 'wb') as f:
        pickle.dump(loglikli, f)
    with open(f"{prefix}dynesty_{jobid}_LIKELIHOODS.pkl", 'wb') as f:
        pickle.dump(mismatch_measurements, f)
    
    print()
    print('Number of posterior samples is {}'.format(len(posterior_samples)))

    corner.corner(
        posterior_samples,
        labels=labels, show_titles=True, color='k'
    )
    
    import pathlib
    pathlib.Path(f"{prefix}figures/{jobid}").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{prefix}figures/{jobid}/corner.pdf")
    print("save", f"{prefix}figures/{jobid}/corner.pdf")

