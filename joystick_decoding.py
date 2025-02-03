import numpy as np
import pandas as pd
import os,sys,glob,inspect,argparse
sys.path.insert(0,'../../')
from TPHATE.tphate import tphate
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr, zscore, percentileofscore
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.ridge import RidgeCV
import analysis_helpers as helper
import nibabel as nib
import mantel
from config import *
from joblib import Parallel, delayed

def run_rsa(X, target_mat):
	Ymat=1-pdist(target_mat,'euclidean') # similarity
	Xmat=1-pdist(X, 'correlation') # correlation, not corr - dist
	mantel_result = mantel.test(Xmat, Ymat, perms=NPERM, method='spearman', tail='upper',ignore_nans=True)
	return mantel_result.r, mantel_result.p, mantel_result.z

def linreg_prediction(X_train, X_test, y_train):
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_hat = linreg.predict(X_test)
    return y_hat

def linear_regression_2d(X_train, X_test, y_train, y_test):
	yhat_xcoord = linreg_prediction(X_train, X_test, y_train[:, 0])
	yhat_zcoord = linreg_prediction(X_train, X_test, y_train[:, 1])
	yhat = np.array((yhat_xcoord, yhat_zcoord)).T
	return mean_squared_error(y_test, yhat)

def get_embedding(subject_id, voxel_data): 
	temp_fn = f'{SCRATCH_PATH}/joystick_analyses/{subject_id}_20d_TPHATE_embedding.npy'
	if os.path.exists(temp_fn):
		tph_dst=np.load(temp_fn)
	if not os.path.exists(temp_fn) or (tph_dst.shape[0] != voxel_data.shape[0]):
		if VERBOSE: print(f'tphate embedding for {subject_id}; n_trs = {voxel_data.shape[0]}')
		tph_dst = tphate.TPHATE(verbose=0, n_components=20, t=8).fit_transform(voxel_data)
		np.save(temp_fn, tph_dst)
	return tph_dst

def load_data_package(subject_id, concat=True, shift_by=2):
	mask_arr = nib.load(f'{DATA_PATH}/{subject_id}/reference/mask.nii.gz').get_fdata()
	data, trial_labels, x_vals, z_vals, run_labels =  [], [], [], [], []
	for run in range(1,5):
		ds, ti, xi, zi = helper.load_all_joystick_data(subject_id, run, mask_arr, shift_by)
		# normalizes the labels within trial
		x_normed = helper.normalize_within_trial(xi, ti)
		z_normed = helper.normalize_within_trial(zi, ti)
		# select only the trial TRs for other data
		in_trials = ti[ti>=0]
		ds_trial = ds[ti>=0]
		data.append(ds_trial)
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
	return np.nan_to_num(data), xs, zs, run_labels, trial_labels

def run_cv_linreg(X, target_mat, cv_labels):
	ps=PredefinedSplit(cv_labels)
	results=np.zeros(ps.get_n_splits())
	for i, (train_index, test_index) in enumerate(ps.split()):
		results[i] = linear_regression_2d(X[train_index], X[test_index], target_mat[train_index], target_mat[test_index])
	return results

def drive_joystick_analyses(subject_id):

	try:
		
		# this data has the labels and brain TRs for all timepoints within trials, shifted by 2 trs, excluding ITI trs
		data, x_coords, z_coords, run_labels, trial_labels = load_data_package(subject_id)
	except:
		print(f'failed loading data {subject_id}')
		nans = [np.nan,np.nan]
		return pd.DataFrame(data={'embedding':nans, 
                                  'subject_id':[subject_id]*2, 
                                  'fold':nans, 
                                  'mse':nans, 
                                  'mantel_z':nans, 
                                  'mantel_rho':nans,'ndim':nans,
                              'zscored_mse':nans})
    
	tph_data = get_embedding(subject_id, data)
	target_coords = np.nan_to_num(np.array((x_coords, z_coords)).T)
	print(f'before clf, tph_nan={np.sum(tph_data!=tph_data)}, data_nan={np.sum(data!=data)}, xnan={np.sum(x_coords!=x_coords)}, znan={np.sum(z_coords!=z_coords)}')
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
		accuracies,zscores = run_cv_linreg_with_null(X, target_coords, run_labels)
		print(embd, accuracies, zscores)
		for i in range(4):
			df.loc[len(df)] = {'embedding':embd, 
                               'subject_id':subject_id, 
                               'fold':i, 
                               'mse':accuracies[i],
                               'mantel_z':z, 
                               'mantel_rho':r, 
                               'ndim':X.shape[1],
                              'zscored_mse':zscores[i]}
	df.to_csv(f'{DATA_PATH}/{subject_id}/results/{subject_id}_joystick_results_jan3.csv')
	print(f'Finished {subject_id}')
	return df


if __name__ == "__main__":
	joblist = []
	joblist = [delayed(drive_joystick_analyses)(S) for S in SUB_IDS]
	with Parallel(n_jobs=16) as parallel:
		results_list = parallel(joblist)
	results_list=pd.concat(results_list).reset_index(drop=True)
	results_list.to_csv(f'{INTERMEDIATE_RESULTS_PATH}/joystick_decoding_results_jan3.csv')


