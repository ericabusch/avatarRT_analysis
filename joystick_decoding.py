import numpy as np
import pandas as pd
import os,sys,glob,inspect,argparse
sys.path.insert(0,'../../')
from TPHATE.tphate import tphate
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr, zscore, percentileofscore
from sklearn.model_selection import PredefinedSplit, KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from himalaya.kernel_ridge import KernelRidgeCV, KernelRidge
from himalaya.ridge import RidgeCV, Ridge
import analysis_helpers as helper
import nibabel as nib
from numpy import linalg
import mantel
from config import *
from joblib import Parallel, delayed

def run_rsa(X, target_mat):
	Ymat=1-pdist(target_mat,'euclidean') # similarity
	Xmat=1-pdist(X, 'correlation') # correlation, not corr - dist
	mantel_result = mantel.test(Xmat, Ymat, perms=NPERM, method='spearman', tail='upper',ignore_nans=True)
	return mantel_result.r, mantel_result.p, mantel_result.z


def himalaya_regression(X_train, X_test, y_train, y_test,inner_cv=True):
    if inner_cv:
        inner_CV = KFold(5)
        alphas= 10.**np.arange(-5, 5, 1)
        reg=RidgeCV(alphas=alphas, cv=inner_CV).fit(X_train, y_train)
    else: 
        reg=Ridge(alpha=0.01).fit(X_train, y_train)
    yhat=reg.predict(X_test)
    mse=mean_squared_error(y_test, yhat)
    return mse

def get_embedding(subject_id, voxel_data,rerun=True): 
	temp_fn = f'{SCRATCH_PATH}/joystick_analyses/{subject_id}_20d_TPHATE_embedding.npy'
	if os.path.exists(temp_fn) and not rerun:
		tph_dst=np.load(temp_fn)
		return tph_dst
	if VERBOSE: print(f'tphate embedding {subject_id}, n_trs = {voxel_data.shape[0]}')
	tph_ds = tphate.TPHATE(verbose=0, n_components=20, t=5).fit_transform(voxel_data)
	np.save(temp_fn, tph_ds)
	return tph_ds

def load_data_package(subject_id, concat=True, shift_by=2):
	mask_arr = nib.load(f'{DATA_PATH}/{subject_id}/reference/mask.nii.gz').get_fdata()
	data, trial_labels, x_vals, z_vals, run_labels =  [], [], [], [], []
	for run in range(1, 5):
		ds, ti, xi, zi = helper.load_all_joystick_data(subject_id, run, mask_arr, shift_by)
		# normalizes the labels within trial
		x_normed = helper.normalize_within_trial(xi, ti)
		z_normed = helper.normalize_within_trial(zi, ti)
		# select only the trial TRs
		in_trials = ti[ti>=0]
		ds_trial = ds[ti>=0]
		data.append(np.nan_to_num(ds_trial))
		trial_labels.append(in_trials)
		x_vals.append(x_normed)
		z_vals.append(z_normed)
		run_labels.append(np.repeat(run, len(in_trials)))
	if concat:
		data=np.concatenate(data, axis=0)
		xs=np.concatenate(x_vals)
		zs=np.concatenate(z_vals)
		run_labels=np.concatenate(run_labels)
		trial_labels=np.concatenate(trial_labels)
	if VERBOSE: print(len(data), len(xs), len(run_labels))
	return data, xs, zs, run_labels, trial_labels

def run_cv_himalaya(X, target_mat, cv_labels, inner_cv=False):
	ps=PredefinedSplit(cv_labels)
	results=[]#np.zeros(ps.get_n_splits())
	for i, (train_index, test_index) in enumerate(ps.split()):
		results.append(himalaya_regression(X[train_index], X[test_index], target_mat[train_index], target_mat[test_index], inner_cv=inner_cv))
	return results


def drive_joystick_analyses(subject_id):

	try:
		# this data has the labels and brain TRs for all timepoints within trials, shifted by 2 trs, excluding ITI trs
        # Try to load, fail if not
		data, x_coords, z_coords, run_labels, trial_labels = load_data_package(subject_id)
	except:
		print(f'failed loading data {subject_id}')
		nans = [np.nan,np.nan]
		return pd.DataFrame(data={'embedding':nans, 
                                  'subject_id':[subject_id]*2, 
                                  'fold':nans, 
                                  'mse':nans, 
                                  'mantel_z':nans, 
                                  'mantel_rho':nans,
                                  'ndim':nans,
                              'zscored_mse':nans})
    
    # Load in the TPHATE embedding if it exists already, else create it
	tph_data = get_embedding(subject_id, data)
	target_coords = np.nan_to_num(np.array((x_coords, z_coords)).T)
	if VERBOSE: print(f'before clf, tph_nan={np.sum(tph_data!=tph_data)}, data_nan={np.sum(data!=data)}, xnan={np.sum(x_coords!=x_coords)}, znan={np.sum(z_coords!=z_coords)}')
	df = pd.DataFrame(columns=['embedding', 
                               'subject_id', 
                               'fold', 
                               'mse', 
                               'mantel_z', 
                               'mantel_rho', 
                               'ndim',
                              'zscored_mse'])

	for X, embd in zip([tph_data, data], [1, 0]):
		r, p, z = run_rsa(X, target_coords)
		accuracies = run_cv_himalaya(X, target_coords, run_labels)
		for i in range(4):
			df.loc[len(df)] = {'embedding':embd, 
                               'subject_id':subject_id, 
                               'fold':i, 
                               'mse':accuracies[i],
                               'mantel_z':z, 
                               'mantel_rho':r, 
                               'ndim':X.shape[1],
                              'zscored_mse':zscores[i]}
	df.to_csv(f'{DATA_PATH}/{subject_id}/results/{subject_id}_joystick_results.csv')
	print(f'Finished {subject_id}')
	return df


if __name__ == "__main__":
	joblist = []
	joblist = [delayed(drive_joystick_analyses)(S) for S in SUB_IDS]
	with Parallel(n_jobs=16) as parallel:
		results_list = parallel(joblist)
	results_list=pd.concat(results_list).reset_index(drop=True)
	results_list.to_csv(f'{INTERMEDIATE_RESULTS_PATH}/joystick_decoding_results_all_subjects.csv')


