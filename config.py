# contains important variables for experiment and analyses
import os, sys, glob
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_PATH = '../..'
DATA_PATH = os.path.expanduser('~/Desktop/BCI/avatarRT_dryad/avatarRT_subject_data')
SCRATCH_PATH = os.path.join(_SCRIPT_DIR, 'results', 'scratch')
PLOTS_PATH = os.path.join(_SCRIPT_DIR, 'results', 'plots')
INTERMEDIATE_RESULTS_PATH = os.path.join(_SCRIPT_DIR, 'results', 'intermediate_results')
FINAL_RESULTS_PATH = os.path.join(_SCRIPT_DIR, 'results', 'final_results')
SESSION_TRACKER = os.path.join(DATA_PATH, 'session_tracker.csv')
SUB_NUMBERS = np.arange(5, 26)
SUB_NUMBERS = SUB_NUMBERS[SUB_NUMBERS!=12] # 12 dropped out
SUB_IDS = [f'avatarRT_sub_{s:02d}' for s in SUB_NUMBERS]
WM_first = [5,7,9,11,13,15,17,19,21,23,24]
WM_FIRST = [f'avatarRT_sub_{s:02d}' for s in WM_first] # subjects who received WMP before OMP
exclude_from_neural_analyses = [12,9,20] # 12 dropped, 9&20 had issues

# Simulated subject codes
CODES=['AA','BB','CC','DD','EE','GG','HH','JJ','KK','MM','NN','PP','QQ','SS','TT','VV','WW','XX','YY','ZZ']
SIMULATED_SUBS = [f'avatarRT_sub_{S}' for S in CODES]

ORIG_MASK=f'{PROJECT_PATH}/ROIs/navigation_mask_MNI_2mm.nii.gz'
SESSION_TYPES_RUNS = {'IM':[1,2,3,4], 'WMP':[2,3,4], 'OMP':[2,3,4]}
ORDER=['IM','WMP','OMP']

SLRAD=3
SHIFTBY=2
SEED=44
REGRESSOR_VERSION='labels'
CALIB_TR=10 # number of calibration TRs excluded from the beginning of each run
NPERM=10000
NBOOT=10000
ALPHAS = 10.**np.arange(-2, 20, 1)
VERBOSE=1

BEHAV_TRIALSERIES = f'behavioral_change_trialseries.csv'
BEHAV_SESSION_RES = f'behavioral_change_session.csv'
BEHAV_TRIAL_W_SIM = f'behavioral_change_trialseries_with_simulations.csv'
BEHAV_SESSION_W_SIM = f'behavioral_change_runwise_with_simulations.csv'

RESULTS_SOURCE = f'{FINAL_RESULTS_PATH}/runwise_neural_variance_control.csv'

colors_main = {'OMP': '#5C940D', 'WMP': '#2E8A82', 'IM': '#00356b'}
colors_sim = {'OMP': '#BCDC8F', 'WMP': '#9ACCD5', 'IM': '#618EBC'}
color_gray = '#A3A3A3'
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




