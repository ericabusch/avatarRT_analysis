#!/bin/bash
#SBATCH --time 6:00:00
#SBATCH --mem-per-cpu 10G -n 2 -c 1 -N 1
#SBATCH --account=turk-browne
#SBATCH --partition=psych_day
#SBATCH --job-name neural_evr_analysis
#SBATCH --output log/%J.out 
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

module load miniconda
conda activate rtcloud_av1

cd /gpfs/milgram/project/turk-browne/users/elb77/BCI/rt-cloud/projects/avatarRT/offline_analyses/final_analysis_scripts/
python -u neural_EVR_analysis.py