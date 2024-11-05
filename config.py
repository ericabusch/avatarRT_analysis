# contains important variables for analyses
import os, sys, glob
import numpy as np

PROJECT_PATH = '/gpfs/milgram/pi/turk-browne/users/elb77/BCI/rt-cloud/projects/avatarRT/'
DATA_PATH = '/gpfs/milgram/pi/turk-browne/users/elb77/BCI/rt-cloud/projects/avatarRT/experiment/subjects'
SCRATCH_PATH='/gpfs/milgram/scratch60/turk-browne/elb77/rtoffline/'
INTERMEDIATE_RESULTS_PATH = f'{PROJECT_PATH}/offline_analyses/final_analysis_scripts/results/intermediate_results'
FINAL_RESULTS_PATH = f'{PROJECT_PATH}/offline_analyses/final_analysis_scripts/results/final_results'

SUB_NUMBERS = np.arange(5, 26)
SUB_NUMBERS = SUB_NUMBERS[SUB_NUMBERS!=12] # 12 dropped out
SUB_IDS = [f'avatarRT_sub_{s:02d}' for s in SUB_NUMBERS]
WM_first = [5,7,9,11,13,15,17,19,21,23,24]
WM_FIRST = [f'avatarRT_sub_{s:02d}' for s in WM_first] # subjects who received WMP before OMP
exclude_from_neural_analyses = [12,9,20]


CODES=['AA','BB','CC','DD','EE','GG','HH','JJ','KK','MM','NN','PP','QQ','SS','TT','VV','WW','XX','YY','ZZ']
SIMULATED_SUBS = [f'avatarRT_sub_{S}' for S in CODES]

ORIG_MASK=f'{PROJECT_PATH}/ROIs/navigation_mask_MNI_2mm.nii.gz'
SESSION_TYPES_RUNS = {'IM':[1,2,3,4], 'WMP':[2,3,4], 'OMP':[2,3,4]}
ORDER=['IM','WMP','OMP']

SLRAD=3
SHIFTBY=2
SEED=44
REGRESSOR_VERSION='regressors_1024'
CALIB_TR=10
NPERM=10000
ALPHAS = 10.**np.arange(-2, 20, 1)
VERBOSE=1

BEHAV_TRIALSERIES = f'{FINAL_RESULTS_PATH}/behavioral_change_trialseries.csv'
BEHAV_SESSION_RES = f'{FINAL_RESULTS_PATH}/behavioral_change_session.csv'



colors_main = {'OMP': '#5C940D', 'WMP': '#2E8A82', 'IM': '#00356b'}
colors_sim = {'OMP': '#BCDC8F', 'WMP': '#9ACCD5', 'IM': '#618EBC'}
context_params = {'font.size': 12.0,
 'axes.labelsize': 'large',
 'axes.titlesize': 'large',
 'xtick.labelsize': 'large',
 'ytick.labelsize': 'large',
 'legend.fontsize': 'medium',
 'legend.title_fontsize': None,
 'axes.linewidth': 0.8,
 'grid.linewidth': 0.8,
 'lines.linewidth': 1.5,
 'lines.markersize': 6.0,
 'patch.linewidth': 2.0,
 'xtick.major.width': 0.8,
 'ytick.major.width': 0.8,
 'xtick.minor.width': 0.6,
 'ytick.minor.width': 0.6,
 'xtick.major.size': 3.5,
 'ytick.major.size': 3.5,
 'xtick.minor.size': 2.0,
 'ytick.minor.size': 2.0}

def does_file_exist(filename):
	b = os.path.isfile(filename)
	if not b: print(f'{filename} DNE')
	return b

# runwise exclusions for original data processing
# all of these runs were replaced 
# EXCLUDE_RUNS = {SUB:{'ses_01':[],'ses_02':[],'ses_03':[],'ses_04':[],'ses_05':[]} for SUB in SUB_IDS}
# EXCLUDE_RUNS['avatarRT_sub_16']['ses_01']+=[2]
# EXCLUDE_RUNS['avatarRT_sub_16']['ses_02']+=[4]
# EXCLUDE_RUNS['avatarRT_sub_16']['ses_03']+=[3]
# EXCLUDE_RUNS['avatarRT_sub_18']['ses_02']+=[1]
# EXCLUDE_RUNS['avatarRT_sub_21']['ses_03']+=[1,2]
# EXCLUDE_RUNS['avatarRT_sub_22']['ses_02']+=[1]
# EXCLUDE_RUNS['avatarRT_sub_23']['ses_02']+=[2]
# EXCLUDE_RUNS['avatarRT_sub_24']['ses_03']+=[1]
# EXCLUDE_RUNS['avatarRT_sub_25']['ses_02']+=[2]




