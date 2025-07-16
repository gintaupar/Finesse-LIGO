# %%
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import corner

# %% Load the samples and combine them all
data = [np.load(_) for _ in glob.glob('samples*.npz')]
samples = defaultdict(list) # combined samples

for _ in data:
    for key in _.keys():
        samples[key].extend(_[key])
        
for key in samples:
    samples[key] = np.array(samples[key])
    
for _ in samples:
    print(_)
    
# %% Select which data to plot
labels = [
    'IM2_Rc',
    'IM3_Rc',
    'PR3_Rc',
    'PR2_Rc',
    'dL[mm]',
    'PRG',
    'PreflPRM',
    'Gouy_PRX',
    'Gouy_PRY',
]

selection = []
for lbl in labels:
    if lbl not in samples:
        raise ValueError(f"Label {lbl} not found in samples")
    if samples[lbl].ndim > 1:
        selection.append(samples[lbl].mean(1))
    else:
        selection.append(samples[lbl])
        
    if "_f" in lbl: # convert to uD
        selection[-1] = 1/selection[-1]/1e-6 

selection = np.array(selection).T
# remove any NaNs
idx = ~np.isnan(selection.mean(1))
selection = selection[idx]

# %%
N = len(labels)
fig, axs = plt.subplots(N, N, figsize=(15, 15))
fig.suptitle('LLO varying PRC geometry\nRed samples are those with P REFL < 80mW for 2W input')
corner.corner(selection, labels=labels, show_titles=True, color='k', fig=fig);

#idx = selection[:, labels.index('PreflPRM')] < 0.08 # Select only the low power samples
idx = np.bitwise_and(
    selection[:, labels.index('Gouy_PRX')] > 18,
    selection[:, labels.index('Gouy_PRX')] < 19,
)

corner.corner(selection[idx, :], fig=fig, color='r');
