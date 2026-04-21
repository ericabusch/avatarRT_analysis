#!/usr/bin/env bash
#SBATCH --output=log/%j-GETDATA.out
#SBATCH --mail-type=all
#SBATCH -p psych_day
#SBATCH -t 1:00:00
#SBATCH --mem 2GB
#SBATCH -n 1
#SBATCH --account=turk-browne

module load XNATClientTools

sess_ID=$1
user=$2
password=$3 

cd ./sandbox/
echo $sess_ID
ArcGet -host https://xnat-milgram.hpc.yale.edu/ -u $user -p $password -s $sess_ID
unzip ${sess_ID}.zip
