#!/bin/bash
#SBATCH --mem-per-cpu 5G -n 2 -c 1 -N 1
#SBATCH --time 10:00:00
#SBATCH --account=turk-browne
#SBATCH --partition=psych_day
#SBATCH --job-name randomise
#SBATCH --output log/%J-randomise.out 
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

. ~/run_rtcloud_setup.sh


INPUT_FN=$1
cd /gpfs/milgram/project/turk-browne/users/elb77/BCI/rt-cloud/projects/avatarRT/offline_analyses/final_analysis_scripts/results/sl_results/
fn=$INPUT_FN
to_replace='.nii.gz'
replace_with='randomise_output'
outfn="${fn/$to_replace/"$replace_with"}"
randomise -i $fn -o $outfn -1 -T -v 5 
