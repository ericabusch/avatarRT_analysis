#!/bin/bash

SUB=$1
N_RUNS=$2

project_dir=./rt-cloud/projects/avatarRT
ROI_FILE=$project_dir/ROIs/zstat_navigation_MNI_2mm.nii.gz
SUB_DIR=$project_dir/experiment/subjects/${SUB}

featdir=$SUB_DIR/ses_01/derivatives/run_01.feat

STD2F=$featdir/reg/standard2example_func.mat

# keep track of reference files
REF_DIR=$SUB_DIR/reference
REF=$REF_DIR/ses_01_example_func.nii.gz


FUNC_DIR=$SUB_DIR/ses_01/func

if [ ! -d $REF_DIR ]; then mkdir $REF_DIR; echo "made $REF_DIR"; fi
if [ ! -d $FUNC_DIR ]; then  mkdir $FUNC_DIR; echo "made $FUNC_DIR"; fi

ROI_OUT=${REF_DIR}/navigation_native.nii.gz
FUNC_STEM=${FUNC_DIR}/${SUB}_task-joystick_

# bring ROI into subject space
flirt -in ${ROI_FILE} -applyxfm -init ${STD2F} -out ${ROI_OUT} -paddingsize 0.0 -interp nearestneighbour -ref ${REF}

echo "Made $ROI_OUT"

# mask with subject brain mask
MASK_FN=${REF_DIR}/ses_01_brain_mask
OUT_FN=${REF_DIR}/navigation_native_brain
fslmaths $ROI_OUT -mul $MASK_FN $OUT_FN

# threshold at 10%
THRP_IMG=${REF_DIR}/navigation_native_brain_thrp25
fslmaths $OUT_FN -thrp 10 $THRP_IMG
echo "Made $THRP_IMG"

# cluster
COUT=${THRP_IMG}_cluster_index_output
SOUT=${THRP_IMG}_cluster_size_output
cluster --zstat=$THRP_IMG -t 2 -o $COUT --osize=$SOUT

# threshold for clusters that are larger than 10 voxels
CROUT=${THRP_IMG}_cluster_mask_size_thresholded.nii.gz
fslmaths -dt int $SOUT -thr 10 -bin $CROUT

cp $CROUT ${REF_DIR}/mask.nii.gz

echo "done ${CROUT}"







