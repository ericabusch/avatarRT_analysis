#!/usr/bin/env bash
#SBATCH --output=log/%j.out
#SBATCH -p psych_week
#SBATCH -t 2:00:00
#SBATCH --mem 4GB
#SBATCH -n 1
#SBATCH --account=turk-browne
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
module load dcm2niix

sess_ID=$1 
echo "running ${sess_ID}"

# set up paths
script_dir="$PWD"
export top_dir=./

export dcm_dir=${top_dir}/sandbox/${sess_ID}/SCANS
export nii_dir=${top_dir}/sandbox/${sess_ID}_nii
export bids_dir=${top_dir}/${project_name}
mkdir -p $nii_dir; cd $dcm_dir

# looping through files in the dicom directories and run 
for k in *
do
    if [ -d "${k}" ]; then
        dcm2niix -o $nii_dir -f %i_%t_%f $dcm_dir/$k
    fi
done
mkdir -p ${bids_dir}; cd ${bids_dir}; mkdir -p raw_data; cd raw_data
