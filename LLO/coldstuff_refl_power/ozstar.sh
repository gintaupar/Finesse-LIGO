#! /bin/bash
#
#SBATCH --job-name=finesse-llo-refl
#
#SBATCH --ntasks=1472
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=800M
#SBATCH --time=12:00:00
export OMP_NUM_THREADS=1 
export PSM2_CUDA=0
srun python ./dynesty_refl.py
python plot_dynesty_refl.py -j $SLURM_JOB_ID
