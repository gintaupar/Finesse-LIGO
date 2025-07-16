#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, pickle, warnings, argparse
from funcs import run_population, model_output_params, reward_params, param_variations

warnings.filterwarnings("ignore")

# HOM max order
MAXTEM = 8
# beam waist parameters
w0 = 1.015e-3
z = 6.0

# Create the parser
parser = argparse.ArgumentParser(description='Genetic Algorithm subPop Run')

# Add arguments
parser.add_argument('--generation_id', type=int, default=10, help='Generation ID')
parser.add_argument('--idstart', type=int, default=0, help='Start ID for the subpopulation to be run')
parser.add_argument('--idend', type=int, default=20, help='End ID for the subpopulation to be run')
parser.add_argument('--thermal_evol_time', type=int, default=5400, help='Time for which thermal model is evolved (sec) (default: 5400)')
parser.add_argument('--cold_evol_time', type=int, default=60, help='Time for which cold model is evolved (sec) (default: 60; a short thermal evolution is as good as cold state solution)')
parser.add_argument('--repopath', type=str, default='.', help='Repository path')
parser.add_argument('--suffix', type=str, default='test', help='Suffix for the output files')

# Parse the arguments
args = parser.parse_args()

gen = args.generation_id
idstart = args.idstart
idend = args.idend

all_sols = []

if 'beam_waist/w0y' in param_variations.keys():
    is_astig = True
else:
    is_astig = False

# load the previous generation
with open(f'{args.repopath}/data/run_{args.suffix}/population_{gen}.pkl', 'rb') as f:
    newpop = pickle.load(f)

# Using beam waist parameters provided by Anamaria
sols, rewards = run_population(newpop,
                               model_output_params,
                               run_thermal=True,
                               idstart=idstart,
                               idend=idend,
                               w0=w0,
                               z=z,
                               maxtems=MAXTEM,
                               is_astig=is_astig,
                               thermal_evol_time=args.thermal_evol_time,
                               cold_evol_time=args.cold_evol_time,
                               reward_params=reward_params,
                               use_real_data=True,
                               )

# slice newpop in the range
for k in newpop:
    newpop[k]['vals'] = newpop[k]['vals'][idstart:idend]

# save the newpop in a pickle file
if not os.path.exists(f'{args.repopath}/data/run_{args.suffix}/gen_{gen}'):
    os.makedirs(f'{args.repopath}/data/run_{args.suffix}/gen_{gen}')

with open(f'{args.repopath}/data/run_{args.suffix}/gen_{gen}/popsols_{gen}_{idstart}_{idend}.pkl', 'wb') as f:
    pickle.dump({'pop': newpop, 'sols': sols, 'rewards': rewards}, f)

print('Finished!')

