# %% [markdown]
# ## Plot results

# %%
import os
os.chdir('/Users/raeddiab/ligo-commissioning-modeling/lho_work/src')

# %%
import os
# from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pickle, glob, tqdm, corner
import warnings
import imageio
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from funcs import (
            get_nested_dict,
            get_actual_params,
            write_yaml_file,
            reward_params,
            param_variations,
            run_thermal_model,
            plot_thermal_evolution,
            )

from thermal_rom import make_ts_optics, add_thermal_detectors

# %%
# suffix = '20240826'
suffix = '20241104'

N = 10000
topX = 0.05
topN = int(topX * N)
parentrepo = "/Users/raeddiab/ligo-commissioning-modeling"
repopath = f"{parentrepo}/lho_work"
# outkeys = ['Px', 'PreflPRM', 'PRG', 'PRG9', 'PRG45', 'gouy']
outkeys = list(reward_params.keys())


if 'beam_waist/w0y' in param_variations.keys():
    is_astig = True
else:
    is_astig = False

w0=1.015e-3
z=6.0

# %%
# check if f'{repopath}/data/run_{suffix}/all_sols.pkl' exists
if False: #os.path.exists(f'{repopath}/data/run_{suffix}/all_sols.pkl'):
    with open(f'{repopath}/data/run_{suffix}/all_sols.pkl', 'rb') as f:
        all_sols = pickle.load(f)
else:
    all_sols = []
    for gen in tqdm.tqdm(range(len(glob.glob(f'{repopath}/data/run_{suffix}/gen_*')))):
        allfiles = glob.glob(f'{repopath}/data/run_{suffix}/gen_{gen}/popsols_{gen}_*.pkl')
        newpop = {}
        sols = {}
        rewards = []
        for f1 in allfiles:
            with open(f1, 'rb') as f:
                dat = pickle.load(f)
                pop, sols1, rewards1 = dat['pop'], dat['sols'], dat['rewards']

                for k in pop:
                    if k not in newpop:
                        newpop[k] = pop[k]
                    else:
                        newpop[k]['vals'] = np.concatenate((newpop[k]['vals'], pop[k]['vals']))

                for k in sols1:
                    if k not in sols:
                        sols[k] = sols1[k]
                    else:
                        sols[k] = np.concatenate((sols[k], sols1[k]))

                rewards = np.concatenate((rewards, rewards1))

        idx = np.argsort(rewards)[::-1][:topN]

        all_sols.append({'sols': sols, 'rewards': rewards, 'topidx': idx, 'newpop': newpop, 'reward_params': reward_params})

        # save all_sols
        with open(f'{repopath}/data/run_{suffix}/all_sols.pkl', 'wb') as f:
            pickle.dump(all_sols, f)

# %%
figfolder = f'{repopath}/figures/figs_{suffix}'
if not os.path.exists(figfolder):
    os.makedirs(figfolder)

# # %%
# xlim = {}
# for ii in range(len(all_sols)):
#     sols, rewards, idx, newpop = (all_sols[ii]['sols'], all_sols[ii]['rewards'], 
#                                   all_sols[ii]['topidx'], all_sols[ii]['newpop'])
#     idx = np.argsort(all_sols[ii]['rewards'])[::-1][:topN]

#     # plot the population
#     plt.figure(figsize=(4,3))
#     plt.plot(np.arange(len(rewards)), rewards, 'o', markersize=1)
#     plt.plot(idx, rewards[idx], 'o', markersize=1)
#     # plt.axhline(8, color='k', linestyle='--')
#     # plt.ylim(0, 6)
#     plt.xlabel('#Solution')
#     plt.ylabel('Reward')
#     plt.show()

#     # plot scatter plots of the solutions
#     fig = plt.figure(figsize=(10, 6))
#     for i, k in enumerate(outkeys):
#         if (k not in sols.keys()) or (len(sols[k]) != len(rewards)):
#             continue
#         plt.subplot(len(reward_params)//3, 3, i+1)
#         plt.scatter(sols[k], rewards, s=1)
#         plt.scatter(sols[k][idx], rewards[idx], s=1)
#         # show vertical shaded region centred at reward_params[k]['centre'] 
#         # and of width reward_params[k]['plateau']
#         plt.axvspan(reward_params[k]['centre']-reward_params[k]['plateu'], 
#                     reward_params[k]['centre']+reward_params[k]['plateu'], 
#                     color='gray', alpha=0.5)
#         # get xlims from current axis
#         if ii == 0: xlim[k] = plt.xlim()
#         plt.xlim(xlim[k])
#         plt.ylim(-20, 15)
#         plt.xlabel(k)
#         plt.ylabel('Reward')
#     plt.tight_layout()
#     plt.savefig(f'{figfolder}/rewards{suffix}_{ii}.png', dpi=100)
#     plt.show()

# %%
# animate the figs
images = []
for i in range(len(all_sols)):
    images.append(imageio.imread(f'{figfolder}/rewards{suffix}_{i}.png'))

imageio.mimsave(f'{figfolder}/rewards{suffix}.gif', images, fps=4, loop=100)
# imageio.mimsave(f'{figfolder}/rewards{suffix}.gif', images, duration=0.1)

# %%
for i in range(3):
    for k in outkeys:
        if (k not in sols.keys()) or ('_rec' in k):
            continue
        print(k, sols[k][idx[i]])
    print()

# %% [markdown]
# ### Custom Corner Plot

# %%
import finesse
from finesse.ligo.factory import ALIGOFactory

def get_actual_params(all_sols,
                      w0=1.015e-3,
                      z=6.0,
                      repopath='/Users/raeddiab/ligo-commissioning-modeling',
                      is_astig=False,
                      ):

    factory = ALIGOFactory(f"{repopath}/LLO/yaml/llo_O4.yaml")
    factory.update_parameters(f"{repopath}/LLO/yaml/llo_addRH.yaml")
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
                                data[ii][path] = k.value + item['vals']
                else:
                    for k in llo.elements['g1'].parameters:
                        if k.name == p[1]:
                            if (wkey in k.name) or (zkey in k.name):
                                data[ii][path] = k.value + item['vals']
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

# %%
data, minmax = get_actual_params(all_sols, w0=w0, z=z, is_astig=is_astig)

# # %%
# for ii in range(len(all_sols)):
#     N1 = len(data[ii].keys())
#     keys = list(data[ii].keys())

#     # Create a figure and a matrix of subplots
#     fig, axs = plt.subplots(N1-1, N1, figsize=(40, 40))  # Adjust figsize as needed

#     # Iterate over the keys to plot each combination
#     for i in range(1,N1):
#         for j in range(N1):
#             ax = axs[i-1, j]
#             if j < i:  # Only keep the lower diagonal
#                 k1, k2 = keys[j], keys[i]
#                 ax.plot(data[ii][k1][idx], data[ii][k2][idx], 'o', markersize=1)
#                 ax.plot(data[ii][k1], data[ii][k2], 'ko', alpha=0.1, markersize=1)
#                 ax.plot(data[ii][k1][idx[:3]], data[ii][k2][idx[:3]], 'ro', markersize=1)
#                 ax.set_xlim(minmax[k1][0], minmax[k1][1])
#                 ax.set_ylim(minmax[k2][0], minmax[k2][1])
#             else:
#                 ax.axis('off')  # Turn off axis for plots not in the upper diagonal

#             # Keep xticks and yticks on the outermost plots
#             if j == 0:  # Leftmost column but only for the upper diagonal
#                 ax.yaxis.set_ticks_position('left')
#                 ax.yaxis.set_label_position('left')
#                 ax.set_yticks(ax.get_yticks())
#                 ax.set_ylabel(k2.replace('PRC/', '').replace('beam_waist/', ' ').replace('/', ' '))
#                 if i == N1-1:
#                     ax.set_xticks(ax.get_xticks())
#                     ax.set_xlabel(k1.replace('PRC/', '').replace('beam_waist/', ' ').replace('/', ' '))
#                 else:
#                     ax.set_xticks([])
#             elif i == N1-1:  # Rightmost column but only for the upper diagonal
#                 # ax.xaxis.set_ticks_position('right')
#                 # ax.xaxis.set_label_position('right')
#                 ax.set_xticks(ax.get_xticks())
#                 ax.set_xlabel(k1.replace('PRC/', '').replace('beam_waist/', ' ').replace('/', ' '))
#                 ax.set_yticks([])
#             else:
#                 ax.set_xticks([])
#                 ax.set_yticks([])

#             # if i == 0 and j > i:  # Set title for the top row but only for the upper diagonal
#             #     ax.set_title(k2)

#     # Adjust layout to prevent overlap
#     plt.tight_layout()
#     plt.savefig(f'{figfolder}/genetic_results{suffix}-PRG_PRG9_Prefl_gouy_{ii}.png', dpi=100)
#     plt.show()

# # make a gif
# images = []
# for ii in range(len(pop)):
#     images.append(imageio.imread(f'{figfolder}/genetic_results_{suffix}-PRG_PRG9_Prefl_gouy_{ii}.png'))

# imageio.mimsave(f'{figfolder}/genetic_results_{suffix}-PRG_PRG9_Prefl_gouy.gif', images, fps=1.2)

# %%
# make a corner plot
# import corner

# for ii in range(len(all_sols)):
#     data1 = data[ii]
#     idx = all_sols[ii]['topidx'][:topN]

#     # Set up the parameters of the problem.
#     ndim, nsamples = len(data1), len(data1)
#     # ndim, nsamples = len(data1), len(idx)

#     cdata = np.vstack([data1[k] for k in data1.keys()]).T
#     # cdata = np.vstack([1e3*data1[k][idx] for k in data1.keys()]).T

#     figure = corner.corner(
#         cdata,
#         labels=[k for k in data1.keys()],
#         plot_datapoints=True,
#         plot_density=False,
#         plot_contours=False,
#         bins=10,
#         density=True,
#         show_titles=True,
#         title_kwargs={"fontsize": 12},
#     )

#     selected_points = cdata[idx, :]

#     # Overplotting
#     num_vars = selected_points.shape[1]
#     for i in range(num_vars):
#         for j in range(num_vars):
#             ax = figure.axes[num_vars * i + j]
#             if i > j:
#                 ax.plot(selected_points[:, j], selected_points[:, i], 'r.', markersize=4)
#                 # ax.set_xlim(minmax[list(data1.keys())[j]])
#                 # ax.set_ylim(minmax[list(data1.keys())[i]])
#             elif i == j:
#                 # Optionally, handle the case for diagonal plots differently
#                 ax.hist(selected_points[:, i], bins=5, color='red', alpha=0.5)
#                 # ax.set_xlim(minmax[list(data1.keys())[i]])
#     plt.tight_layout()

#     plt.savefig(f'{figfolder}/genetic_results{suffix}_ip_params_{ii}.png', dpi=100)
#     plt.show()

# %%
# # suffix = '20240809'
# figfolder = f'{repopath}/figures/figs_{suffix}'

# # animate the figs
# images = []
# for i in range(len(all_sols)):
#     images.append(imageio.imread(f'{figfolder}/genetic_results{suffix}_ip_params_{i}.png'))

# imageio.mimsave(f'{figfolder}/genetic_results{suffix}_ip_params.gif', images, fps=4, loop=100)

# %% [markdown]
# ## Write select states to yaml files

# %%
ii = len(all_sols)-1

lastgendata = data[ii]
nesteddict = get_nested_dict(lastgendata)

sols, rewards, idx, newpop = all_sols[ii]['sols'], all_sols[ii]['rewards'], all_sols[ii]['topidx'], all_sols[ii]['newpop']
idx = np.argsort(all_sols[ii]['rewards'])[::-1][:topN]

# select regions of interest
# isel = np.where(((sols['PRG'][idx] - sols['PRG_thermalised'][idx]) > 0) & ((sols['PRG9'][idx] - sols['PRG9_thermalised'][idx]) < 0))[0][-2:-1]
isel = np.arange(5, dtype=int)
# isel[0] = np.where(sols['PreflPRM'][idx] < 0.0525)[0][0]
# isel[1] = np.where(sols['PreflPRM'][idx] > 0.0569)[0][0]
# isel[2] = np.where(sols['PRG9'][idx] < 87.95)[0][0]
# isel[3] = np.where(sols['PRG9'][idx] > 88.5)[0][0]
# isel[4] = np.where(sols['PRG'][idx] > 32)[0][0]

# plot scatter plots of the solutions
fig = plt.figure(figsize=(10, 6))
for i, k in enumerate(outkeys):
    if (k not in sols.keys()) or (len(sols[k]) != len(rewards)):
        continue
    plt.subplot(3, 3, i+1)
    # plt.scatter(sols[k], rewards, s=1)
    plt.scatter(sols[k][idx], rewards[idx], s=1)
    plt.scatter(sols[k][idx[isel]], rewards[idx[isel]], s=1)
    # plt.ylim(-0.1, 6.5)
    plt.xlabel(k)
    plt.ylabel('Reward')
plt.tight_layout()
plt.show()

# %%
dataout = {}
for i, k in enumerate(outkeys):
    if '_rec' in k:
        continue
    dataout[k] = sols[k][idx]

ii = len(all_sols)-1
sols, rewards, idx, newpop = all_sols[ii]['sols'], all_sols[ii]['rewards'], all_sols[ii]['topidx'], all_sols[ii]['newpop']
idx = np.argsort(all_sols[ii]['rewards'])[::-1][:topN]

# create a corner plot for dataout dictionary
ndim, nsamples = len(dataout), len(dataout['PRG'])
cdata = np.vstack([dataout[k] for k in dataout.keys()]).T

figure = corner.corner(
    cdata,
    labels=[k for k in dataout.keys()],
    plot_datapoints=True,
    plot_density=False,
    plot_contours=False,
    bins=10,
    density=True,
    show_titles=True,
    title_kwargs={"fontsize": 9},
)

figure.set_size_inches(20,20)

selected_points = cdata[isel, :]

# Overplotting
num_vars = selected_points.shape[1]
for i in range(num_vars):
    for j in range(num_vars):
        ax = figure.axes[num_vars * i + j]
        if i > j:
            ax.plot(selected_points[:, j], selected_points[:, i], 'r.', markersize=4)
            # get axis limits and set them 20% larger
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xdiff = np.abs(xlim[1] - xlim[0])
            ydiff = np.abs(ylim[1] - ylim[0])
            ax.set_xlim(xlim[0]-0.2*xdiff, xlim[1]+0.2*xdiff)
            ax.set_ylim(ylim[0]-0.2*ydiff, ylim[1]+0.2*ydiff)
        elif i == j:
            # Optionally, handle the case for diagonal plots differently
            ax.hist(selected_points[:, i], bins=5, color='red', alpha=0.5)
            # get axis limits and set them 20% larger
            xlim = ax.get_xlim()
            xdiff = np.abs(xlim[1] - xlim[0])
            ax.set_xlim(xlim[0]-0.2*xdiff, xlim[1]+0.2*xdiff)

plt.show()

# %%
yamlfolder = f'{repopath}/data/run_{suffix}/cold_state_yaml_solutions'
if not os.path.exists(yamlfolder):
    os.makedirs(yamlfolder)

print("Writing the cold state mode matching solutions")
for i in range(isel.shape[0]):
    yamlfile = f'{yamlfolder}/cold_state_sol_{i}.yaml'
    _ = write_yaml_file(nesteddict, idx[isel][i], yamlfile, print_text=False)
    print(f'Written {yamlfile}')
    # print('****'*10)

# %%
lastgendata = data[ii]

for k, v in lastgendata.items():
    print(k, v[idx[isel][0]], newpop[k]['vals'][idx[isel][0]])

# %%
for k in newpop:
    print(f"'{k}': {{'vals': [{newpop[k]['vals'][idx[isel][0]]}], 'type': '{newpop[k]['type']}'}},")

# %%
# for k, v in lastgendata.items():
#     min1, max1 = newpop[k]['vals'][idx].min(), newpop[k]['vals'][idx].max()
#     d = max1 - min1
#     print(k, v[idx[isel][0]], np.round(min1-d, 6), np.round(max1+d, 6))

# %% [markdown]
# # Tests

# %% [markdown]
# ### Comparing yaml loading vs param var loading

# %%
# yaml loading
import glob
import os

from funcs import run_thermal_model, plot_thermal_evolution, model_output_params, reward_params
from thermal_rom import *
from funcs import *

# suffix = '20241104'
repopath = "/Users/raeddiab/ligo-commissioning-modeling/lho_work"
yamls = glob.glob(f'{repopath}/data/run_{suffix}/cold_state_yaml_solutions/*.yaml')
yamls.sort()

figfolder = f'{repopath}/figures/figs_{suffix}'
os.makedirs(figfolder, exist_ok=True)

factory = ALIGOFactory("/Users/raeddiab/ligo-commissioning-modeling/LLO/yaml/llo_O4.yaml")
factory.update_parameters("/Users/raeddiab/ligo-commissioning-modeling/LLO/yaml/llo_addRH.yaml")
factory.update_parameters(yamls[0])
if 'absorption' in factory.params:
    tm_abs = factory.params.absorption

factory=None
maxtems=8
RH_efficiency={'ITMX': 0.9, 'ITMY': 0.9, 'ETMX': 0.9, 'ETMY': 0.9}
waist_params={}
losses={}
reward_params={}
datapath=f'{repopath}/data/run_{suffix}/cold_state_yaml_solutions/thermal_evolution_{suffix}_{i}.pkl'
return_data_level=2
update_yaml=yamls[0]
w0=1.015e-3
z=6.0
t_evol=6000
runsols = {}
model_output_params=model_output_params
tm_absorption=tm_abs
is_astig=True

# %%
if not factory:
    factory = ALIGOFactory("/Users/raeddiab/ligo-commissioning-modeling/LLO/yaml/llo_O4.yaml")
    factory.update_parameters("/Users/raeddiab/ligo-commissioning-modeling/LLO/yaml/llo_addRH.yaml")

if update_yaml:
    factory.update_parameters(update_yaml)

factory.params.INPUT.LASER.power = 2

# Make the model
factory.reset() # always reset to default
factory.options.LSC.add_output_detectors = True
factory.options.ASC.add = True
# factory.options.INPUT.add_IMC_and_IM1 = True

# add all apertures
factory.options.apertures.add = True
factory.options.apertures.PR3_SR3 = True
factory.options.apertures.BS_ITM = True
factory.options.apertures.BS_HR = True

ts_itmx, ts_etmx, ts_itmy, ts_etmy = make_ts_optics()

llo = factory.make()

if update_yaml:
    llo = set_cold_state_beam(llo, w0, z, update_factory=factory, waist_params={}, losses={}, is_astig=is_astig)
else:
    llo = set_cold_state_beam(llo, w0, z, update_factory=None, waist_params=waist_params, losses=losses, is_astig=is_astig)

add_thermal_detectors(llo)

# ring heater powers [W] with 70% efficiency from requested power to optic
P_RH_ITMX = factory.params.P_RH_ITMX * RH_efficiency['ITMX']
P_RH_ITMY = factory.params.P_RH_ITMY * RH_efficiency['ITMY']
P_RH_ETMX = factory.params.P_RH_ETMX * RH_efficiency['ETMX']
P_RH_ETMY = factory.params.P_RH_ETMY * RH_efficiency['ETMY']

def set_ringheaters(arm, P_RH_ITM, P_RH_ETM):
    lens = llo.get(f"ITM{arm}lens")
    lens.f = 1 / (1 / lens.f + P_RH_ITM * factory.params.IRH_sub)
    itm = llo.get(f"ITM{arm}")
    etm = llo.get(f"ETM{arm}")
    itm.Rc = 2 / (2 / itm.Rc + P_RH_ITM * factory.params.IRH_srf)
    etm.Rc = 2 / (2 / etm.Rc + P_RH_ETM * factory.params.ERH_srf)
    print(f"New values for ITM{arm} lens f: {lens.f.value}, ITM{arm} Rc: {itm.Rc}, ETM{arm} Rc: {etm.Rc}")

set_ringheaters("X", P_RH_ITMX, P_RH_ETMX)
set_ringheaters("Y", P_RH_ITMY, P_RH_ETMY)

# %%



