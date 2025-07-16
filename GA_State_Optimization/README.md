# Finesse LIGO State Optimisation

The goal of this work is to find a state of [finesse ligo](https://finesse.docs.ligo.org/finesse-ligo/index.html) that matches the real LLO observations. A genetic algorithm is used to match observed data.

Sample usage
```
export suffix=$(date +%Y%m%d)
repopath=$(pwd)

mkdir -p ${repopath}/job_files/jobs_${suffix}
mkdir -p ${repopath}/logs/logs_${suffix}
mkdir -p ${repopath}/data/run_${suffix}

./src/generate_dag.py --total_generations 20 --ids_per_simulation 10 --population_size 1000 --job_file_dir ${repopath}/job_files/jobs_${suffix} --repopath ${repopath} --suffix ${suffix} --top_x_percent 0.1 --frac_crossover 0.3 --frac_mutation 0.8 --frac_shrink 0.7 --thermal_evol_time 40 --cold_evol_time 40
```
[Migrated from [finesse-ligo-state-optimisation](https://git.ligo.org/shreejit.jadhav/finesse-ligo-state-optimisation)]

For any details, contact shreejit.jadhav@ligo.org
