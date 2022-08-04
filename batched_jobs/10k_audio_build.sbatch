#!/bin/bash

#SBATCH --job-name=10kaud
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=100:00:00
#SBATCH --mail-user=chris.ick@nyu.edu
#SBATCH --export=NONE
#SBATCH --output="o_10k_aud_build-%j.out"

module purge
module load anaconda3/2020.07

source /home/ci411/.bashrc
conda activate s3d_env
python /home/ci411/volume_estimation/batched_jobs/10k_audio_build.py