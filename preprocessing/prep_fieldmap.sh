#!/bin/bash
#Prepare a fieldmap for FEAT
#Based on https://lcni.uoregon.edu/kb-articles/kb-0003 & Neuropipe

#SBATCH --output=log/%j-topup.out
#SBATCH -p psych_day
#SBATCH -t 300
#SBATCH --mem 4G
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --account=turk-browne
#SBATCH --job-name topup

module load FSL/6.0.5-centos7_64 ;
. /gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.5-centos7_64/etc/fslconf/fsl.sh;

SUBJECT_ID=$1  # avatarRT_sub-01
SESSION_ID=$2 # ses-01

# create intermediate file dir
BASEDIR=./rt-cloud/projects/avatarRT/experiment/subjects
THISDIR=./rt-cloud/projects/avatarRT/preprocessing
FMAP_DIR=$BASEDIR/${SUBJECT_ID}/${SESSION_ID}/fmap/

# get the two SE images and concatenate them 
AP_FN=$FMAP_DIR/${SUBJECT_ID}_DistortionMap_AP.nii
PA_FN=$FMAP_DIR/${SUBJECT_ID}_DistortionMap_PA.nii
CONCAT_FN=$FMAP_DIR/all_SE.nii.gz
fslmerge -t $CONCAT_FN $AP_FN $PA_FN

# Create the acquisition file
# First three columns are PE direction, fourth column is total readout time (can be found in the header).
# j- is 0 -1 0; j is 0 1 0
PARAMS=$FMAP_DIR/acqparams.txt
cat > $PARAMS << EOF
0 -1 0 0.01638
0 1 0 0.01638
EOF
echo "starting topup"

topup --imain=$CONCAT_FN --datain=$PARAMS --config=$THISDIR/odd_b02b0.cnf --out=$FMAP_DIR/topup_output --iout=$FMAP_DIR/topup_iout --fout=$FMAP_DIR/topup_fout --logout=$FMAP_DIR/topup_logout

echo "finished topup"

# topup_iout are the unwarped SE images; check these to see if unwarping went well. use the average of these images as magnitude image (see below).
# topup_fout is the fieldmap, in Hz.
# Convert fieldmap from Hz to rad/s:

FMAP_RAD=$FMAP_DIR/fieldmap_rads.nii.gz
fslmaths $FMAP_DIR/topup_fout -mul 6.28 $FMAP_RAD
FMAP_RAD_BRAIN=$FMAP_DIR/fieldmap_rads_brain.nii.gz
bet $FMAP_RAD ${FMAP_RAD_BRAIN}  -B -f 0.25

#Create the magnitude image & BET it
FMAP_MAG=$FMAP_DIR/fieldmap_magnitude.nii.gz
fslmaths $FMAP_DIR/topup_iout -Tmean $FMAP_MAG
FMAP_MAG_BRAIN=$FMAP_DIR/fieldmap_magnitude_brain.nii.gz
bet $FMAP_MAG $FMAP_MAG_BRAIN -B -f 0.25 -g 0

# erode
fslmaths $FMAP_MAG_BRAIN -ero $FMAP_MAG_BRAIN

echo "Finished topup; final files at $FMAP_MAG_BRAIN"