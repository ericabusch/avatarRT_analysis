import numpy as np
import sys, glob
sys.path.append('..')
from subprocess import call
import nibabel as nib
import os, sys, argparse

# first run sbatch get_xnat.sh {ses_ID} {xnatPword} (avatarRT_sub_01_ses_01 Avatar4!}
# then convert to niftis: dicom2nii2bids.sh {ses_ID} {avatarRT_sub_01_ses_01}


################ Parse CL args #######################
parser = argparse.ArgumentParser()
parser.add_argument("-sub", "--subject_id",type=str, help="avatarRT_sub_XX")
parser.add_argument("-ses", "--session_number", type=int, help="session number")
args = parser.parse_args()
sub_ID = args.subject_id
ses_num = args.session_number
ses_ID = f'ses_{ses_num:02d}'
session = f'{sub_ID}_{ses_ID}'
print(f' --> Running {session}')
################ done parsing CL args #######################
valid_scans = [1,2,3,4,5,6,7,8,9,10,14,18]
print(f'Confirm these scan numbers: {valid_scans}')

scan_types = ['AA_scout', 
'AAHead_Scout_64ch-head-coil_MPR_sag', 
'AAHead_Scout_64ch-head-coil_MPR_cor', 
              'AAHead_Scout_64ch-head-coil_MPR_tra',
              'joystick','joystick','joystick','joystick','DistortionMap_AP',
'DistortionMap_PA', 'T1w', 'T2w']
scan_labels=['na','na','na','na','func','func','func','func','fmap','fmap','anat','anat']

MATCHES = len(valid_scans) == len(scan_types) == len(scan_labels)
assert MATCHES

print(f"All labels match? {MATCHES}")

verbose=True
project_directory = './rt-cloud/projects/avatarRT/'
subj_dir = f'{project_directory}/experiment/subjects/{sub_ID}'

#create file structure in this directory for this subject
os.makedirs(subj_dir, exist_ok=True)
print(f'Creating {subj_dir}')
os.makedirs(f'{subj_dir}/reference', exist_ok=True)
for s in range(1,6):
    os.makedirs(f'{subj_dir}/ses_{s:02d}', exist_ok=True)
    os.makedirs(f'{subj_dir}/ses_{s:02d}/func', exist_ok=True)
    os.makedirs(f'{subj_dir}/ses_{s:02d}/anat', exist_ok=True)
    os.makedirs(f'{subj_dir}/ses_{s:02d}/fmap', exist_ok=True)
    os.makedirs(f'{subj_dir}/ses_{s:02d}/scripts', exist_ok=True)
print('created dir structure')

sandbox='/gpfs/milgram/scratch60/turk-browne/elb77/sandbox/'
outdir = f'{sandbox}/{session}_nii'
subj_dir = f'{project_directory}/experiment/subjects/{sub_ID}'


files = []
for scan in valid_scans:
    print(f'{outdir}/{session}_*_{scan}.nii')
    filename = glob.glob(f'{outdir}/{session}_*_{scan}.nii')[0]
    files.append(filename.replace('.nii',''))

print(f'have {len(files)} to copy')
func_run = 0
for i, scan, fn in zip(np.arange(len(valid_scans)), valid_scans, files):
    if scan_labels[i] is 'func':
        func_run += 1
        for suffix in ['json','nii']:
            outfn = f'{subj_dir}/ses_01/func/{sub_ID}_task-{scan_types[i]}_run-{func_run:02d}_bold.{suffix}'
            print(fn, outfn)
            infn = fn+f'.{suffix}'
            command = f'cp {infn} {outfn}'
            call(command, shell=True)

    elif scan_labels[i] is 'anat':
        for suffix in ['json', 'nii']:
            outfn = f'{subj_dir}/ses_01/anat/{sub_ID}_{scan_types[i]}.{suffix}'
            infn = fn + f'.{suffix}'
            command = f'cp {infn} {outfn}'
            call(command, shell=True)

    elif scan_labels[i] == 'fmap':
        for suffix in ['json', 'nii']:
            outfn = f'{subj_dir}/ses_01/fmap/{sub_ID}_{scan_types[i]}.{suffix}'
            print(fn, outfn)
            infn = fn + f'.{suffix}'
            command = f'cp {infn} {outfn}'
            call(command, shell=True)

    else:
        print(f'not copying {fn} of type {scan_types[i]}')


# # run bet on the anatomical scans
for fn in glob.glob(f'{subj_dir}/ses_01/anat/*.nii'):
    outfn = fn.replace('.nii','_brain.nii')
    command = f"bet {fn} {outfn} -f 0.5 -g 0"
    call(command, shell=True)

# figure out the number of timepoints per volume
n_timepoints=[]
for fn in sorted(glob.glob(f'{subj_dir}/{ses_ID}/func/*.nii')):
    img = nib.load(fn)
    n_timepoints.append(img.shape[-1])
    print(f'n_voxels={img.shape[0]*img.shape[1]*img.shape[2]}')

with open(f'{subj_dir}/{ses_ID}/scripts/timing.txt','w') as f:
    for n in n_timepoints:
        f.write(f'{n}\n')
    print(f"Made the timing file: {n_timepoints}")
                                           
                                           
                                           
                                           
                                           
                                           
                                           
