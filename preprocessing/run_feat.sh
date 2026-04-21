#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 16
#SBATCH --mem-per-cpu 1G
#SBATCH --time 12:00:00
#SBATCH --account=turk-browne
#SBATCH --partition=psych_day
#SBATCH --job-name FEAT
#SBATCH --output log/%J-feat.out 
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

module load FSL/6.0.5-centos7_64 ;
. /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.5-centos7_64/etc/fslconf/fsl.sh ;

DESIGN_SCRIPT=$1
echo $DESIGN_SCRIPT 
feat $DESIGN_SCRIPT
echo "Done feat"

