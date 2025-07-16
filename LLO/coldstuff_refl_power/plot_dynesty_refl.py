# %%
import corner
import numpy as np
import finesse
from scipy.stats import truncnorm, norm, uniform
import numpy as np
import matplotlib.pyplot as plt
import finesse
import numpy as np
import finesse
import matplotlib.pyplot as plt
from corner.core import quantile
import dill as pickle
from dynesty.utils import resample_equal
import pathlib
        
        
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

finesse.init_plotting(fmts=['png'])

prefix = '/fred/oz147/'

# %%
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-j", "--jobid", default=None, dest="jobid", help="Slurm job ID to process result of")

(options, args) = parser.parse_args()
time = options.jobid

# %%
pathlib.Path(f"{prefix}figures/{time}").mkdir(parents=True, exist_ok=True)

# %%
rank = 0
file = f"{prefix}dynesty_{rank}_{time}_samples.npz"
data = np.load(file)
posterior_samples = data['posterior_samples']
labels = data['labels']

with open(f"{prefix}dynesty_{rank}_{time}_prior_transform.pkl", "rb") as f:
    prior_transform = pickle.load(f)
    
with open(f"{prefix}dynesty_{rank}_{time}_likelihood.pkl", "rb") as f:
    loglikli = pickle.load(f)
    
with open(f"{prefix}dynesty_{rank}_{time}_LIKELIHOODS.pkl", "rb") as f:
    LIKELIHOODS = pickle.load(f)


# get priors
N = 100000
u = np.zeros((posterior_samples.shape[1], N))
u[:] = np.random.uniform(0, 1, N)[None, :]
x = prior_transform(u)

# %%
ndim = data['labels'].size
fig = corner.corner(
    posterior_samples,
    labels=labels,
    show_titles=True,
    color='k', 
    range=[(a.min(), a.max())for a in posterior_samples.T],
);
axes = np.array(fig.axes).reshape(ndim, ndim)

nom_PRM = -11.009
nom_PR2 = -4.548
nom_PR3 = 36.027

values = [
    nom_PRM,
    nom_PR2,
    nom_PR3,
    0,
    0,
    70,
    0,
    1,
    5.945,
    0.7,
    0.7,
    5e-6,
]
ax = axes[0, 0]
ax.axvline(nom_PR2, color="r")
ax = axes[1, 1]
ax.axvline(nom_PR3, color="r")

# Loop over the histograms
for yi in range(ndim):
    for xi in range(yi):
        if values[xi] is not None and values[yi] is not None:
            ax = axes[yi, xi]
            ax.axvline(values[xi], color="r")
            ax.axhline(values[yi], color="r")
            ax.plot(values[xi], values[yi], "sr")
     
        
for i in range(ndim):
    plt.sca(axes[i, i]);
    axes[i,i].clear()
    axes[i,i].get_yaxis().set_visible(False)
    
    plt.hist(
        x[i],
        20,
        density=True,
        alpha=0.5,
        color='c',
    )
    plt.hist(
        posterior_samples[:, i],
        20,
        density=True,
        alpha=0.8,
        color='k',
        fill=False,
        histtype='step',
    )
    if values[i] is not None:
        axes[i,i].axvline(values[i], color="r")
    
    a, b, c = quantile(posterior_samples[:, i], [0.16, 0.5, 0.84])
    plt.title(fr"{labels[i]}=${{{b:.2f}}}^{{{b-a:.3f}}}_{{{b-c:.3f}}}$", fontsize=9)
    
plt.savefig(f"{prefix}figures/{time}/corner_detailed_{time}.pdf")

# %%
for key in data['blob'].dtype.fields:
    plt.figure()
    Y = data['blob'][key]    
    Y = resample_equal(Y, data['weights'])
    _, bins, _ = plt.hist(Y, 50, density=True, label='Posterior', alpha=0.7, histtype='stepfilled');
    if key in LIKELIHOODS:
        plt.hist(LIKELIHOODS[key][0].ppf(np.random.rand(Y.size), **LIKELIHOODS[key][1]), bins, density=True, label='Measurement', alpha=0.7, histtype='stepfilled');
    plt.xlabel(key)
    plt.ylabel("Probability density")
    plt.title(f"Posterior for {key}")    
    plt.legend()
    plt.savefig(f"{prefix}figures/{time}/blob_histograms_{key}_{time}.pdf")