import numpy as np
import os, sys, glob
import shutil, argparse
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument("-sub", "--subject_id",type=str, help="avatarRT_sub_XX")
parser.add_argument("-ses", "--session_number", type=int, help="session number")
parser.add_argument("-nr", "--num_runs", type=int, help='number of runs in this session')
args = parser.parse_args()
sub_ID = args.subject_id
ses_num = args.session_number
ses_ID = f'ses_{ses_num:02d}'
session = f'{sub_ID}_{ses_ID}'
n_runs = args.num_runs
print(f' --> Reorganize after running FEAT for {session}, {n_runs} runs')

tasknames = ['joystick']*n_runs
project_directory = './rt-cloud/projects/avatarRT/'
subj_dir = f'{project_directory}/experiment/subjects/{sub_ID}'

dirnames = [os.path.join(subj_dir, ses_ID, 'derivatives', f'run_{r:02d}.feat') for r in range(1,n_runs+1)]

for dirr, run in zip(dirnames, range(1,n_runs+1)):
    funcvol = os.path.join(dirr, 'filtered_func_data.nii.gz')
    newname = os.path.join(subj_dir, ses_ID, 'func', f'{sub_ID}_task-{tasknames[run-1]}_run-{run:02d}_bold_preproc.nii.gz')
    shutil.copy(funcvol, newname)
    print(newname)
dirr=dirnames[0]
mean_func = os.path.join(dirr, 'mean_func.nii.gz')
mask = os.path.join(dirr, 'mask.nii.gz')
out = os.path.join(subj_dir, 'reference', 'mean_func_brain.nii.gz')
mask_out = os.path.join(subj_dir, 'reference', 'ses_01_brain_mask.nii.gz')
command = f'fslmaths {mean_func} -mas {mask} {out}'
call(command, shell=True)
print(command)
shutil.copy(mask, mask_out)
shutil.copy(out, os.path.join(subj_dir, 'reference', 'ses_01_example_func.nii.gz'))


    
