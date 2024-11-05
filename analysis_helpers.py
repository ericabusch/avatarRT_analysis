import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import sys, os, glob, json
import random
from random import choices, choice
from config import *
from sklearn.preprocessing import StandardScaler
from functools import reduce
import nibabel as nib
sys.path.insert(0,'../..')
from MRAE.mrae import ManifoldRegularizedAutoencoder
from MRAE import dataHandler
from TPHATE.tphate import tphate

######## functions pertaining to getting experiment information

def flatten_trials(trials):
    trials1 = trials.copy()
    diffs = np.where(np.diff(trials)<0)[0]
    for i, d in enumerate(diffs):
        to_add = trials[diffs[i]]+1
        for k in range(d+1, len(trials)):
            trials1[k] = trials1[k]+to_add
    assert np.all(np.diff(trials1)<=1)
    return trials1

def format_subid(subject_number):
    try: sub=f'avatarRT_sub_{subject_number:02d}'
    except: sub=f'avatarRT_sub_{subject_number}'
    return sub

def get_perturbation_info(subject_id, session_id, run, return_component=False):
    if run == 1: 
        if return_component: return 'IM', 0
        else: return 'IM'
    
    df = pd.read_csv(f'{PROJECT_PATH}/offline_analyses/info/session_tracker_amended.csv')
    sub_num = str(int(subject_id.split('_')[-1]))
    ses_num = int(session_id.split('_')[-1])
    row = df[df['subject_number'] == sub_num]
    if row.omp_session.item() == ses_num: 
        perturb_type = "OMP"
    elif row.im_session.item() == ses_num:
        perturb_type = "IM"
    else: 
        perturb_type = "WMP"
    if not return_component:  return perturb_type
    if perturb_type == 'OMP': comp = int(row.omp_component.item())
    elif perturb_type == 'WMP': comp = int(row.wmp_component.item())
    else: comp = 0
    return perturb_type, comp

def load_info_file():
	return pd.read_csv(f'{PROJECT_PATH}/offline_analyses/info/session_tracker_amended.csv')

def get_success_rate_file(subject_id):
	df = pd.read_csv(f'{DATA_PATH}/{subject_id}/results/success_rates.csv')
	df['brain_control']=1-df['game_control']
	return df

def get_brain_control_from_success_file(df, session, run, trial=None):
    if trial!=None:
        row = df[(df['session_number']==session) & (df['run_number']==run) & (df['round_number'].isin(trial))]
        return row.brain_control.values, row.index
    rows = df[(df['session_number']==session) & (df['run_number']==run)]['brain_control'].values
    return np.nanmean(rows)

######### data processing & statistics

def permutation_test(data, n_iterations, alternative='greater'):
    np.random.seed(0)
    random.seed(0)
    """
    permutation test for comparing the means of two distributions 
    where the samples between the two distributions are paired
    
    """
    
    def less(null_distribution, observed):
        cmps = null_distribution <= observed + gamma
        pvalues = (cmps.sum(axis=0) + 1) / (n_iterations + 1)
        return pvalues

    def greater(null_distribution, observed):
        cmps = null_distribution >= observed - gamma
        pvalues = (cmps.sum(axis=0) + 1) / (n_iterations + 1)
        return pvalues

    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues
    
    compare = {'less': less, 'greater': greater, 'two-sided': two_sided}
    n_samples = data.shape[1]
    observed_difference = data[0] - data[1]
    observed = np.mean(observed_difference)
    
    
    null_distribution = np.empty(n_iterations)
    for i in range(n_iterations):
        weights = [choice([-1, 1]) for d in range(n_samples)]
        null_distribution[i] = (weights*observed_difference).mean()
        
    eps = 1e-14
    gamma = np.maximum(eps, np.abs(eps * observed))
    
    pvalue = compare[alternative](null_distribution, observed)
    return observed, pvalue, null_distribution

def normalize(X, axis=0):
	# normalize function for standard implementation across scripts
    X = zscore(X, axis=axis)
    return np.nan_to_num(X)

def run_EVR(X_transformed):
    X_var = np.var(X_transformed, axis=0)
    evr = X_var / np.sum(X_var)
    return np.array(evr)

def embed_tphate(X, t=5, n_components=2):
    embd = tphate.TPHATE(t=t, n_components=n_components, verbose=0).fit_transform(X)
    return embd

####### functions for wrangling data

def extract_data_from_df(dataframe, filter_columns, filter_values, statistic=None, return_index=False):
    ## takes a dataframe, filters columns to specific values, returns statistics
    indices = []
    for c,v in zip(filter_columns, filter_values):
        if type(v) == str:
            idx = list(dataframe.query(f'{c} == "{v}"').index)
        else:
            idx = list(dataframe.query(f"{c} == {v}").index)
        indices.append(idx)
    overlap_indices = reduce(np.intersect1d, indices)
    if statistic:
        return dataframe.iloc[overlap_indices][statistic].values
    if return_index: return dataframe.iloc[overlap_indices]
    return dataframe.iloc[overlap_indices].reset_index() 

def aggregate_files(to_match, output_fn):
    fns = sorted(glob.glob(to_match))
    df=pd.concat([pd.read_csv(f,index_col=0) for f in fns])
    try:
        df.drop(labels=['index'],axis=1,inplace=True)
    except:
        if VERBOSE: print(df.columns, 'boop')
    df.reset_index(drop=True).to_csv(output_fn)
    if VERBOSE: print(f'saved to {output_fn}')
    return df

def shift_timing(nTRs_data, label_TR, TR_shift_size, start_label=0):
    toTrimData=0
    # Create a short vector of extra values - whatever you want
    start_shift = np.ones((TR_shift_size,)) * start_label
    # Pad the column from the top.
    label_TR_shifted = np.concatenate((start_shift, label_TR))
    # Check what has been shifted off the timeline
    if len(label_TR_shifted) > nTRs_data:
        label_TR_shifted = label_TR_shifted[0:nTRs_data]
    elif len(label_TR_shifted) < nTRs_data:
        toTrimData = nTRs_data - len(label_TR_shifted)
    else:
        toTrimData = 0
    return label_TR_shifted,toTrimData

def get_trial_data(X, subject_id, session_id, run, shift_by=2):
    '''
    takes a timeseries matrix for a given run and separates those timepoints into trials, including BOLD shift

    X = pre-loaded data matrix, [n_trs, nfeatures]
    subject_id, session_id, run = information used to load regressors
    shift_by = amount to shift labels for BOLD lag
    '''
    # altered from parse_data_by_regressor
    reg_df = pd.read_csv(f'{DATA_PATH}/{subject_id}/{REGRESSOR_VERSION}/{subject_id}_{session_id}_run_{run:02d}_timeseries_regressors.csv', index_col=0)
    trial_OnOff = reg_df['isTrial'].values
    trial_numbers = reg_df['trial'].values
    trial_OnOff_shifted, toTrim = shift_timing(X.shape[0], trial_OnOff, shift_by, start_label=0)
    trial_label_shifted, toTrim = shift_timing(X.shape[0], trial_numbers, shift_by, start_label=-1)
    
    while len(trial_OnOff_shifted) > len(reg_df):
        reg_df.loc[len(reg_df)] = {"TR":len(reg_df)+1, "TR_0idx":len(reg_df),  "trial":np.nan,  "isTrial":0}
        if VERBOSE: print('appending to reg df')

    while len(trial_OnOff_shifted) < len(reg_df):
        trial_OnOff_shifted = np.append(trial_OnOff_shifted,[0])
        trial_label_shifted = np.append(trial_label_shifted,[-1])
        if VERBOSE: print('appending to labels ')

    reg_df['TR-0idx'] = reg_df['TR_0idx'] 
    reg_df['isTrial_shifted'] = trial_OnOff_shifted
    reg_df['trial_shifted'] = trial_label_shifted
    trial_TRs = reg_df[reg_df['isTrial_shifted'] == 1][['TR-0idx', 'trial_shifted']] # where are the trials after shifting
    
    trial_numbers = sorted(trial_TRs['trial_shifted'].unique())
    data_by_trial ={}
    for trial in trial_numbers:
        trs_in_trial = np.array(trial_TRs[trial_TRs['trial_shifted'] == trial]['TR-0idx'].values, dtype=int)
        trs_in_trial = trs_in_trial[(trs_in_trial >= 0) & (trs_in_trial < X.shape[0])]
        d = X[trs_in_trial]
        if d.shape[0] < 2:  continue
        data_by_trial[trial] = d
    if VERBOSE: print(f"After getting trial data, found {len(trial_numbers)} trials, data shape: {len(data_by_trial)}, Xshape: {X.shape}")
    return data_by_trial

def list_by_trial(X, trial_labels, normalize):
    X_list=[]
    for t in sorted(np.unique(trial_labels)):
        if t == -1: 
            continue
        trial_idx = np.where(trial_labels==t)[0]
        if X.ndim == 4: 
            X_trial = X[...,trial_idx]
        else: 
            X_trial=X[trial_idx]
        if normalize: X_trial=normalize(X_trial)
        X_list.append(X_trial)
    return X_list

def load_regressor_dfs(subject_id, session_id, run):
    df1 = pd.read_csv(f'{DATA_PATH}/{subject_id}/{REGRESSOR_VERSION}/{subject_id}_{session_id}_run_{run:02d}_timeseries_regressors.csv', index_col=0)
    df2 = pd.read_csv(f'{DATA_PATH}/{subject_id}/{REGRESSOR_VERSION}/{subject_id}_{session_id}_run_{run:02d}_trial_regressors.csv', index_col=0)
    return df1,df2

####### functions pertaining to loading brain data of different sorts

def get_realtime_outdata(subject_id, session_id, run_number, data_type=None):
    # returns just one type of data
    if data_type:
        return np.load(f"{DATA_PATH}/{subject_id}/{session_id}/data/{subject_id}_run_{run_number:02d}_{data_type}.npy")
    # returns a dictionary of all types
    to_return = {}
    for data_type in ['masked_data','masked_raw_data','preproc_times','projected_data','analysis_times','decoded_angles','embedded_timeseries']:
        to_return[data_type] =  np.load(f"{DATA_PATH}/{subject_id}/{session_id}/data/{subject_id}_run_{run_number:02d}_{data_type}.npy")
    return to_return

def get_realtime_data_preprocesssed(subject_id, session_id, run, data_type=None):
	# offline preprocessing of real time data
    return np.load(f"{DATA_PATH}/{subject_id}/{session_id}/data_rt_offline_preproc/{subject_id}_run_{run:02d}_masked_data_v2.npy")

def load_vol_data(subject_id, session_id, run_ID, space='standard', asarray=False, normalize=True):
    task = 'RT'
    root = f'{DATA_PATH}/{subject_id}/{session_id}/func/{subject_id}_task-{task}_run-{run_ID:02d}'
    if session_id == 'ses_01': task='joystick'
    if space == 'standard': tail = '_bold_preproc_v2_denoised_MNI152_2mm.nii.gz'
    elif space == 'native': tail = '_bold_preproc_v2_denoised_native.nii.gz'
    elif space == 'fmriprep': tail = '_space-MNI152Lin_desc-preproc_fmriprep_bold.nii.gz'
    else: tail = '_bold.nii'
    fn = f'{root}{tail}'
    X = nib.load(fn)
    if asarray: return X.get_fdata()
    return X

def load_manifold_projected_data(subject_id, session_id, run,by_trial=False,shift_by=2):
    '''
    Returns the manifold embedding of the data in real-time
    '''
    X =  get_realtime_outdata(subject_id, session_id, run, 'projected_data')
    if by_trial: 
        X = get_trial_data(X, subject_id, session_id, run, shift_by=shift_by)
    else:
        X=X[CALIB_TR:]
    return X

def load_component_data(subject_id, session_id, run, component_number=None, by_trial=False, shift_by=2):
    # get projection onto NFB components
    X = calculate_nfb_component_loadings(subject_id, session_id, run, component_number=component_number)
    if by_trial: 
        X = get_trial_data(X, sub_ID, ses_ID, run, shift_by)
    else:
        X=X[CALIB_TR+shift_by:]
    return X

def calculate_nfb_component_loadings(subject_id, session_id, run, component_number=None):
    '''
    Returns the loading of brain data onto either specific components of the neurofeedback or all 
    '''
    modelPath=f'{DATA_PATH}/{subject_id}/model'
    X = get_realtime_outdata(subject_id, session_id, run, 'masked_data') # this is the real-time normalized data
    if component_number:
        this_mrae, _, _, NFB_component = load_model_from_dir(modelPath, perturbation=component_number, verbose=0)
        X_proj = this_mrae.extract_projection_to_manifold(X)
        X_mapped = map_projection(X_proj, NFB_component)
    else:
        this_mrae, _, _, _ = load_model_from_dir(modelPath, perturbation=1, verbose=0)
        X_proj = this_mrae.extract_projection_to_manifold(X)
        components = get_manifold_component(this_mrae.manifold_regularization)
        X_mapped = map_projection(X_proj, components)
    return X_mapped

# define the component of the manifold
def get_manifold_component(manifold, n_components=None):
    pca = PCA(n_components=n_components).fit(manifold)
    return pca.components_

# project a new array x into the fitted MRAE's manifold layer and extract the projection
def project_new_data(mrae_model, x):
    xDataset = dataHandler.TestDataset(x)
    xProj = mrae_model.extract_projection_to_manifold(xDataset)
    return xProj

# map projected data onto the fitted PC
def map_projection(projected_data, fitted_component):
    loading = np.dot(projected_data, fitted_component.T)
    return loading

def load_model_from_dir(modelPath, perturbation=0, verbose=False):
    modelFilename = f'{modelPath}/state_dict.pt'
    modelSpec = f'{modelPath}/modelSpec.txt'
    bottleneck = np.load(f'{modelPath}/bottleneck.npy')
    
    if int(perturbation) == 0:
        if verbose: print("Loading intrinsic mapping")
        MANI_COMP = np.load(f'{modelPath}/manifold_pc_01.npy')
        TEST_RANGE = np.load(f'{modelPath}/test_range_01.npy') 
            
    elif int(perturbation) == 1:
        if verbose: print("Loading on-manifold perturbation")
        MANI_COMP = np.load(f'{modelPath}/manifold_pc_02.npy') 
        TEST_RANGE = np.load(f'{modelPath}/test_range_02.npy') 

    else:
        if verbose: print(f"Loading off-manifold perturbation; component={perturbation}")
        MANI_COMP = np.load(f'{modelPath}/manifold_pc_{perturbation:02d}.npy') 
        TEST_RANGE = np.load(f'{modelPath}/test_range_{perturbation:02d}.npy') 
    
    # this file contains the 1st and 99th percentile loadings of val data onto the pc of the AE bottleneck layer
    # get the slope and intercept for the mapping from manifold component to direction
    MAPPING_SLOPE, MAPPING_INTERCEPT = np.polyfit(TEST_RANGE, [-1, 1], 1)
    # set the mapping function
    with open(modelSpec, 'r') as f:
        modelParams = json.load(f)

    # initialize model
    this_mrae = ManifoldRegularizedAutoencoder(hidden_dim=int(modelParams['hidden_dim']),
                                                    manifold_dim=int(modelParams['manifold_dim']),
                                                    IO_dim=int(modelParams['IO_dim']))
    this_mrae.load_model_state_dict(modelFilename) # load in the pretrained weights
    this_mrae.manifold_regularization = bottleneck # pass bottleneck
    return this_mrae, MAPPING_SLOPE, MAPPING_INTERCEPT, MANI_COMP


def load_joystick_task_location_labels(subject_id, run, nTRs_data, shift_by=2):
    reg_df = pd.read_csv(f'{DATA_PATH}/{subject_id}/ses_01/behav/run_{run:03d}_events_master_revised_v2.csv', index_col=0)
    trial_OnOff = reg_df['isTrial'].values
    trial_numbers = reg_df['trial_number'].values
    xvalues = reg_df['x_coord_norm'].values
    zvalues = reg_df['z_coord_norm'].values
    trial_numbers_shifted, toTrim = shift_timing(nTRs_data, trial_numbers, shift_by, start_label=-1)
    xvalues_shifted,_ = shift_timing(nTRs_data, xvalues, shift_by, start_label=0)
    zvalues_shifted,_ = shift_timing(nTRs_data, zvalues, shift_by, start_label=0)
    return trial_numbers_shifted, xvalues_shifted, zvalues_shifted,toTrim

def load_all_joystick_data(subject_id, run, mask, shift_by=2):
    infn = f'{DATA_PATH}/{subject_id}/ses_01/func/{subject_id}_task-RT_run-{run:02d}_bold_preproc_v2_native.nii.gz'
    outfn= f'{SCRATCH_PATH}/joystick_analyses/{subject_id}_run_{run:02d}_bold_preproc_v2_native_navigation_mask.npy'
    if os.path.exists(outfn):
        ds = np.load(outfn)
        if VERBOSE: print(f"loading {outfn} of shape {ds.shape}")
    else:
        nii_data=nib.load(infn).get_fdata()
        ds = nii_data[mask==1].T
        ds = normalize(ds, axis=0)
        np.save(outfn, ds)
        if VERBOSE: print(f"saved {outfn} of shape {ds.shape}")
    trials_shifted, xshifted, zshifted,toTrim = load_joystick_task_location_labels(subject_id, run, ds.shape[0], shift_by)
    if VERBOSE: print(f"data,trials,xshifted,zshifted of shape {ds.shape},{trials_shifted.shape},{xshifted.shape},toTrim={toTrim}")
    if toTrim != 0: ds = ds[:-1*toTrim] 
    return ds, trials_shifted, xshifted, zshifted # trim the bold data

######## functions pertaining to unity measures

def load_location_labels(subject_id, session_id, run, nTRs_data, shift_by=2, version=2):
    reg_df = pd.read_csv(f'{DATA_PATH}/{subject_id}/{REGRESSOR_VERSION}/{subject_id}_{session_id}_run_{run:02d}_timeseries_regressors.csv', index_col=0)
    xvalues = reg_df['x_coord_norm'].values
    zvalues = reg_df['z_coord_norm'].values
    trial_OnOff = reg_df['isTrial'].values
    trial_numbers = reg_df['trial'].values
    
    trial_numbers_shifted, toTrim = shift_timing(nTRs_data, trial_numbers, shift_by, start_label=0)
    xvalues_shifted,_ = shift_timing(nTRs_data, xvalues, shift_by, start_label=0)
    zvalues_shifted,_ = shift_timing(nTRs_data, zvalues, shift_by, start_label=0)
    
    return trial_numbers_shifted, xvalues_shifted, zvalues_shifted

def euclidean_distance(coords, target_coord):
    diffs = coords - target_coord
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))
    return distances

def get_distance_traveled(subject_id, session_id, run, trial, threshold=0.008):
    fn = f"{DATA_PATH}/{subject_id}/{session_id}/behav/run_{run:03d}/round_{trial:02d}/player_transform.txt"
    transf = pd.read_csv(fn)
    transf = transf[transf["Event"]=="Transform.Position"]
    x_vec, z_vec = transf.x.values, transf.z.values
    delta = 0 
    for i in range(1,len(x_vec)):
        x0, z0 = float(x_vec[i-1]), float(z_vec[i-1])
        x1, z1 = float(x_vec[i]), float(z_vec[i])
        dist = np.sqrt((x0-x1)**2 + (z0-z1)**2)
        if dist > threshold: delta+=dist
    return delta

def get_shortest_path_length(subject_id, session_id, run, trial):
    fn = f"{DATA_PATH}/{subject_id}/{session_id}/behav/run_{run:03d}/round_{trial:02d}/player_transform.txt"
    transf = pd.read_csv(fn)
    transf = transf[transf["Event"]=="Transform.Position"]
    x0, z0 = float(transf.x.values[0]), float(transf.z.values[0] )
    x1, z1 = float(transf.x.values[-1]), float(transf.z.values[-1])
    pathlen = np.sqrt((x0-x1)**2 + (z0-z1)**2)
    return pathlen

def get_trial_error(subject_id, session_id, run, trial):
    delta = get_distance_traveled(subject_id, session_id, run, trial)
    pathlen = get_shortest_path_length(subject_id, session_id, run, trial)
    return (delta-pathlen)/pathlen

def get_target_location(subject_id, session_id, run, trial):
    fn = f"{DATA_PATH}/{subject_id}/{session_id}/behav/run_{run:03d}/round_{trial:02d}/gameboard.txt"
    board = pd.read_csv(fn)
    tar = board[board["ObjectType"]=="Target"]
    tar.x = tar.x.astype(float)
    tar.z = tar.z.astype(float)
    x, z = tar.x.values[-1], tar.z.values[-1]
    return x, z

def get_start_location_unity(subject_id, session_id, run, trial):
    fn = f"{DATA_PATH}/{subject_id}/{session_id}/behav/run_{run:03d}/round_{trial:02d}/player_transform.txt"
    transf = pd.read_csv(fn)
    transf = transf[transf["Event"]=="Transform.Position"]
    x, z = transf.x.values[0], transf.z.values[0]
    return x,z

def get_position_file(subject_id, session_id, run, trial):
    fn = f"{DATA_PATH}/{subject_id}/{session_id}/behav/run_{run:03d}/round_{trial:02d}/player_transform.txt"
    transf = pd.read_csv(fn)
    transf = transf[transf["Event"]=="Transform.Position"]
    return transf

def get_end_location_unity(subject_id, session_id, run, trial):
    fn = f"{DATA_PATH}/{subject_id}/{session_id}/behav/run_{run:03d}/round_{trial:02d}/player_transform.txt"
    transf = pd.read_csv(fn)
    transf = transf[transf["Event"]=="Transform.Position"]
    x, z = transf.x.values[-1], transf.z.values[-1]
    return x,z




