#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle, warnings, argparse, glob
from funcs import sample_param_variations, mutation_crossover, param_variations, np

warnings.filterwarnings("ignore")

# Create the parser
parser = argparse.ArgumentParser(description='Genetic Algorithm Configuration')

# Add arguments
parser.add_argument('--seed', type=int, default=420, help='Random seed')
parser.add_argument('--population_size', type=int, default=20, help='Size of the population')
parser.add_argument('--generation_id', type=int, default=0, help='Generation ID')
parser.add_argument('--top_x_percent', type=float, default=0.3, help='Top X% best solutions as parents')
parser.add_argument('--frac_crossover', type=float, default=0.5, help='Fraction for crossover')
parser.add_argument('--frac_mutation', type=float, default=0.8, help='Fraction for mutation')
parser.add_argument('--frac_shrink', type=float, default=0.5, help='Fraction of mutation size to shrink by')
parser.add_argument('--repopath', type=str, default='.', help='Repository path')
parser.add_argument('--suffix', type=str, default='test', help='Suffix for the output files')

# Parse the arguments
args = parser.parse_args()

# Use the arguments
np.random.seed(args.seed)

N = args.population_size
gen = args.generation_id
topX = args.top_x_percent
frac_crossover = args.frac_crossover
frac_mutation = args.frac_mutation
frac_shrink = args.frac_shrink

topN = int(topX * N)

# Run the genetic algorithm
if gen == 0:
    # Sampling for the first generation
    newpop = param_variations

    _ = sample_param_variations(newpop, N)
else:
    # load the previous generation data
    allfiles = glob.glob(f'{args.repopath}/data/run_{args.suffix}/gen_{gen-1}/popsols_{gen-1}_*.pkl')
    newpop = {}
    sols = {}
    rewards = []
    for f in allfiles:
        with open(f, 'rb') as f:
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

    # shrink the mutation range
    for k in newpop:
        newpop[k]['var'] *= frac_shrink

    # get the new population
    idx = np.argsort(rewards)[::-1][:topN]
    newpop = mutation_crossover(newpop, topN, N, idx)

# save the newpop in a pickle file
with open(f'{args.repopath}/data/run_{args.suffix}/population_{gen}.pkl', 'wb') as f:
    pickle.dump(newpop, f)

print('Finished!')

