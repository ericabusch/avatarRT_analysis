set -e

SUBJ=$1
SES=$2
N_RUNS=$3
 

BASEDIR=./rt-cloud/projects/avatarRT/experiment/subjects/${SUBJ}/${SES}/
ROOT=./rt-cloud/projects/avatarRT/experiment/subjects/${SUBJ}/

FEAT_DIR=${BASEDIR}/derivatives/
mkdir -p $FEAT_DIR

function get_value( ) {
  TARGET=$1
  n=1
  while read line;
  do
    if test $n -eq $TARGET
    then
     VOLS=$line
    fi
    n=$((n+1))
  done < $TIMING_FN
}

RAW_SUBJ_DIR=${BASEDIR}
FUNCDIR=${RAW_SUBJ_DIR}/func
T1W_FN=${ROOT}/ses_01/anat/${SUBJ}_T1w_brain.nii.gz
FMAP_UNWARP_FN=${RAW_SUBJ_DIR}/fmap/fieldmap_rads.nii.gz
FMAP_MAG_FN=${RAW_SUBJ_DIR}/fmap/fieldmap_magnitude_brain.nii.gz

# find timing file
TIMING_FN=${RAW_SUBJ_DIR}/scripts/timing.txt
if [ -s $TIMING_FN ]
then
echo "Found timing file at " $TIMING_FN
fi

for RUN in `seq 1 $N_RUNS`
do
  OUTPUT_DIR=${FEAT_DIR}/run_0${RUN}
  RUN_FN=`ls ${FUNCDIR}/${SUBJ}_task-*_run-0${RUN}_bold.nii` 
  echo $RUN_FN
  get_value "$RUN"
  cp feat_design.fsf.template tempDesign.fsf
  T=tempDesign.fsf
  sed -i "s:<?= \$OUTPUT_DIR ?>:$OUTPUT_DIR:g" $T 
  sed -i "s:<?= \$N_VOLUMES ?>:$VOLS:g" $T 
  sed -i "s:<?= \$DATA_FILE_PREFIX ?>:$RUN_FN:g" $T 
  sed -i "s:<?= \$STRUCT_IMG ?>:$T1W_FN:g" $T
  sed -i "s:<?= \$FIELDMAP_MAG_FILE ?>:$FMAP_MAG_FN:g" $T
  sed -i "s:<?= \$FIELDMAP_UNWARP_FILE ?>:$FMAP_UNWARP_FN:g" $T

  echo "Finished SED to " $T "for subject,run " $SUBJ $RUN

  DESIGN_FN=${RAW_SUBJ_DIR}/scripts/${SUBJ}_run-0${RUN}_feat_design.fsf
  mv tempDesign.fsf $DESIGN_FN
  echo "RENDERED AT " $DESIGN_FN
  sbatch run_feat.sh $DESIGN_FN
  echo "RUNNING $DESIGN_FN"
done