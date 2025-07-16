#! /usr/bin/env python3

import argparse
import textwrap

Parser = argparse.ArgumentParser(description="Generate DAG file for HTCondor")
Parser.add_argument("--total_generations", type=int, help="Total number of generations", required=True)
Parser.add_argument("--ids_per_simulation", type=int, help="Number of IDs per simulation", required=True)
Parser.add_argument("--population_size", type=int, help="Population size", required=True)
Parser.add_argument("--job_file_dir", type=str, help="Job file directory", required=True)
Parser.add_argument("--repopath", type=str, default=".", help="Repository path")
Parser.add_argument("--suffix", type=str, default="test", help="Suffix for the run files")

# Sampling arguments
Parser.add_argument('--top_x_percent', type=float, default=0.3, help='Top X% best solutions as parents')
Parser.add_argument('--frac_crossover', type=float, default=0.5, help='Fraction for crossover')
Parser.add_argument('--frac_mutation', type=float, default=0.8, help='Fraction for mutation')
Parser.add_argument('--frac_shrink', type=float, default=0.5, help='Fraction of mutation size to shrink by')

args = Parser.parse_args()

def generate_sub_file(sub_file_path, repopath, executable, argument_comb, suffix, job_suffix):
    content = textwrap.dedent(f"""\
    Universe = vanilla
    Executable = {repopath}/src/{executable}.sh
    Arguments = {argument_comb}
    Output = {repopath}/logs/logs_{suffix}/{executable.strip('run_')}_{suffix}{job_suffix}.out.$(Cluster).$(Process)
    Error = {repopath}/logs/logs_{suffix}/{executable.strip('run_')}_{suffix}{job_suffix}.err.$(Cluster).$(Process)
    Log = {repopath}/logs/logs_{suffix}/{executable.strip('run_')}_{suffix}{job_suffix}.log.$(Cluster).$(Process)

    requirements = ( OpSys == "LINUX" ) && (Machine != "node2397.cluster.ldas.cit") && (Machine != "node2394.cluster.ldas.cit")
    request_memory = 16384M
    request_disk = 4096M
    request_cpus = 1
    accounting_group = ligo.sim.o4.detchar.explore.test
    notification = never
    getenv = true
    Should_transfer_files = Yes
    +MaxHours = 240
    x509userproxy = /tmp/x509up_u46160

    Queue

    """)

    with open(sub_file_path, 'w') as f:
        f.write(content)

def generate_dag_file(total_generations, ids_per_simulation, population_size, dag_file_path, top_x_percent=0.3, frac_crossover=0.5, frac_mutation=0.8, frac_shrink=0.5, repopath=".", suffix="test"):
    with open(dag_file_path, 'w') as f:
        for generation in range(total_generations):
            # Write the sampling job for the current generation
            f.write(f"JOB JobSampling{generation} {args.job_file_dir}/submit_sampling_{suffix}.sub\n")
            f.write(f"VARS JobSampling{generation} generation_id=\"{generation}\" population_size=\"{population_size}\" top_x_percent=\"{top_x_percent}\" frac_crossover=\"{frac_crossover}\" frac_mutation=\"{frac_mutation}\" frac_shrink=\"{frac_shrink}\" repopath=\"{repopath}\" suffix=\"{suffix}\"\n\n")

            # Calculate the number of simulation jobs based on ids_per_simulation
            num_sim_jobs = (population_size + ids_per_simulation - 1) // ids_per_simulation

            # Write the simulation jobs for the current generation
            for sim_job in range(num_sim_jobs):
                start_id = sim_job * ids_per_simulation
                end_id = min(start_id + ids_per_simulation, population_size)
                job_name = f"JobRunsims{generation}_{sim_job + 1}"
                f.write(f"JOB {job_name} {args.job_file_dir}/submit_runsims_{suffix}.sub\n")
                f.write(f"VARS {job_name} generation_id=\"{generation}\" idstart=\"{start_id}\" idend=\"{end_id}\" repopath=\"{repopath}\" suffix=\"{suffix}\"\n")
            f.write("\n")

            # Define dependencies
            sim_jobs_names = [f"JobRunsims{generation}_{sim_job + 1}" for sim_job in range(num_sim_jobs)]
            if generation == 0:
                f.write(f"PARENT JobSampling{generation} CHILD {' '.join(sim_jobs_names)}\n\n")
            else:
                f.write(f"PARENT {' '.join(prev_sim_jobs_names)} CHILD JobSampling{generation}\n")
                f.write(f"PARENT JobSampling{generation} CHILD {' '.join(sim_jobs_names)}\n\n")
            prev_sim_jobs_names = sim_jobs_names

# write submit files
generate_sub_file(f"{args.job_file_dir}/submit_sampling_{args.suffix}.sub", args.repopath, "run_GA_sampling", "$(generation_id) $(population_size) $(top_x_percent) $(frac_crossover) $(frac_mutation) $(frac_shrink) $(repopath) $(suffix)", args.suffix, '_$(generation_id)')

generate_sub_file(f"{args.job_file_dir}/submit_runsims_{args.suffix}.sub", args.repopath, "run_GA_runsims", "$(generation_id) $(idstart) $(idend) $(repopath) $(suffix)", args.suffix, '_$(generation_id)_$(idstart)_$(idend)')

# write dag file
dagfilepath = f"{args.job_file_dir}/dag_submit_{args.suffix}.dag"
generate_dag_file(args.total_generations, args.ids_per_simulation, args.population_size, dagfilepath, top_x_percent=args.top_x_percent, frac_crossover=args.frac_crossover, frac_mutation=args.frac_mutation, frac_shrink=args.frac_shrink, repopath=args.repopath, suffix=args.suffix)

print(f"DAG file saved as {dagfilepath}")
