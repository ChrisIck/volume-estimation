#!/bin/bash

#SBATCH --job-name=trBL-ARL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=47:59:59
#SBATCH --mail-user=chris.ick@nyu.edu
#SBATCH --export=NONE
#SBATCH --output="o_trBL-ARL-%j.out"
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2020.07

source /home/ci411/.bashrc
conda activate s3d_env
python /home/ci411/volume_estimation/batched_jobs/trBL-ARL.py