#! /bin/bash
#
#SBATCH --job-name=finesse-llo-single_bounce
#
#SBATCH --ntasks=1472
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=800M
#SBATCH --time=2:00:00
export OMP_NUM_THREADS=1 
export PSM2_CUDA=0
srun python ./dynesty_IMC_OMC.py --dlogz=0.1 --prefix=/fred/oz147/LLO_single_bounce --nlive=2000
python plot_dynesty_IMC_OMC.py -j $SLURM_JOB_ID --prefix=/fred/oz147/LLO_single_bounce