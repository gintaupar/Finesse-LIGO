# Finesse LIGO State Optimisation

The goal of this work is to find a state of [finesse ligo](https://finesse.docs.ligo.org/finesse-ligo/index.html) that matches the real LLO observations. A genetic algorithm is being used to find the cold state first. The aim is to match thermal evolution ultimately.

Sample usage
```
export suffix=$(date +%Y%m%d)
repopath=/home/shreejit.jadhav/WORK/RC/ligo-commissioning-modeling/finesse-ligo-state-optimisation

mkdir -p ${repopath}/job_files/jobs_${suffix}
mkdir -p ${repopath}/logs/logs_${suffix}
mkdir -p ${repopath}/data/run_${suffix}

./src/generate_dag.py --total_generations 20 --ids_per_simulation 5 --population_size 10000 --job_file_dir ${repopath}/job_files/jobs_${suffix} --repopath ${repopath} --suffix ${suffix} --top_x_percent 0.05 --frac_crossover 0.5 --frac_mutation 0.8 --frac_shrink 0.5
```

For any details, contact shreejit.jadhav@ligo.org
