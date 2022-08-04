#!/bin/bash

#SBATCH --job-name=trP1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=47:59:59
#SBATCH --mail-user=chris.ick@nyu.edu
#SBATCH --export=NONE
#SBATCH --output="o_trP1-%j.out"
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2020.07

source /home/ci411/.bashrc
conda activate s3d_env
python /home/ci411/volume_estimation/batched_jobs/trP1.py