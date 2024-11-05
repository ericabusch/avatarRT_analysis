import numpy as np
import pandas as pd
import os,sys,glob,inspect,argparse
sys.path.insert(0,'../../')
from TPHATE.tphate import tphate
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr, zscore
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import LinearRegression
from himalaya.kernel_ridge import KernelRidgeCV
from himalaya.ridge import RidgeCV
import analysis_helpers as helper
import nibabel as nib
import mantel
from config import *
from joblib import Parallel, delayed

def run_rsa(X, target_mat):
	Ymat=1-pdist(target_mat,'euclidean')
	Xmat=1-pdist(X, 'correlation')
	mantel_result = mantel.test(Xmat, Ymat, perms=NPERM, method='spearman', tail='upper')
	return mantel_result.r, mantel_result.p, mantel_result.z

def linear_regression_2d(X_train, X_test, y_train, y_test):
	mod1 = LinearRegression()
	mod1.fit(X_train, y_train[:,0])
	pred1= mod1.predict(X_test)
	mod2 = LinearRegression()
	mod2.fit(X_train, y_train[:,1])
	pred2= mod1.predict(X_test)
	y_hat = np.array((pred1, pred2)).T
	dis = np.linalg.norm(y_hat-y_test, axis=1)
	return np.mean(dis)

def get_embedding(subject_id, voxel_data):
	temp_fn = f'{SCRATCH_PATH}/joystick_analyses/{subject_id}_20d_TPHATE_embedding.npy'
	if os.path.exists(temp_fn):
		tph_dst=np.load(temp_fn)
	if not os.path.exists(temp_fn) or (tph_dst.shape[0] != voxel_data.shape[0]):
		tph_dst = tphate.TPHATE(verbose=0, n_components=20, t=8).fit_transform(voxel_data)
		np.save(temp_fn, tph_dst)
	return tph_dst

def load_data_package(subject_id, concat=True, shift_by=2):
	mask_arr = nib.load(f'{DATA_PATH}/{subject_id}/reference/mask.nii.gz').get_fdata()
	data, trial_labels, x_vals, z_vals, run_labels, analysis_mask = [], [], [], [], [], []
	for run in range(1,5):
		ds, ti, xi, zi = helper.load_all_joystick_data(subject_id, run, mask_arr, shift_by)
		data.append(ds)
		trial_labels.append(ti)
		x_vals.append(xi)
		z_vals.append(zi)
		run_labels.append(np.repeat(run, len(ds)))
		analysis_mask.append(np.array([int(t >= 0) for t in ti]))
	if concat:
		data=np.concatenate(data, axis=0)
		xs=np.concatenate(x_vals)
		zs=np.concatenate(z_vals)
		run_labels=np.concatenate(run_labels)
		trial_labels=np.concatenate(trial_labels)
		analysis_mask = np.concatenate(analysis_mask)
	if VERBOSE: print(len(data), len(xs), len(analysis_mask))
	return np.nan_to_num(data), xs, zs, run_labels, trial_labels, analysis_mask

def run_cv_linreg(X, target_mat, cv_labels):
	ps=PredefinedSplit(cv_labels)
	results=np.zeros(ps.get_n_splits())
	for i, (train_index, test_index) in enumerate(ps.split()):
		results[i] = linear_regression_2d(X[train_index], X[test_index], target_mat[test_index], target_mat[test_index])
	return results

def randomize_labels(labels, perm=0):
	if perm==0: return labels
	return np.random.permutation(labels)

def run_cv_linreg_with_null(X, target_mat, cv_labels):
	ps=PredefinedSplit(cv_labels)
	results=np.zeros((NPERM, ps.get_n_splits())) # will hold zscores now
	for i, (train_index, test_index) in enumerate(ps.split()):
		if VERBOSE: print(X[train_index].shape, X[test_index].shape, target_mat[test_index].shape)
		results[0, i] = linear_regression_2d(X[train_index], X[test_index], target_mat[train_index], target_mat[test_index])
		for p in range(1, NPERM):
			permuted_labels = randomize_labels(target_mat, perm=p)
			results[p, i] = linear_regression_2d(X[train_index], X[test_index], target_mat[train_index], permuted_labels[test_index])
	true_accuracies = results[0, :]
	zscores = zscore(results, axis=1)[0]		
	return true_accuracies, zscores

def run_ridge_cv_himalaya(X, target_mat, cv_labels):
	outer_split = PredefinedSplit(cv_labels)
	results=np.zeros(outer_split.get_n_splits())
	for i, (train_index, test_index) in enumerate(outer_split.split()):
		inner_cv_labels = cv_labels[train_index]
		inner_CV = PredefinedSplit(inner_cv_labels)
		if X.shape[0] > X.shape[1]:
			model = RidgeCV(alphas=ALPHAS, cv=inner_CV).fit(X[train_index], target_mat[train_index])
		else:
			model = KernelRidgeCV(alphas=ALPHAS, cv=inner_CV, kernel='linear').fit(X[train_index], target_mat[train_index])
		predicted_targets = model.predict(X[test_index])
		results[i] = np.mean(np.linalg.norm(predicted_targets-target_mat[test_index], axis=1))
	return results

def drive_joystick_analyses(subject_id):
	try:
		data, x_coords, z_coords, run_labels, trial_labels, analysis_mask = load_data_package(subject_id)
	except:
		return df = pd.DataFrame(data={'embedding':np.nan, 'subject_id':subject_id, 'fold':np.nan, 
			'accuracy_zscore':np.nan, 'raw_accuracy':np.nan, 'mantel_z':np.nan, 
		'mantel_rho':np.nan, 'accuracy_himalaya':np.nan})
	tph_data = get_embedding(subject_id, data)
	target_coords = np.array((x_coords, z_coords)).T
	
	target_coords = target_coords[analysis_mask==1]
	tph_data = tph_data[analysis_mask==1]
	data = data[analysis_mask==1]
	run_labels=run_labels[analysis_mask==1] - 1

	df = pd.DataFrame(columns=['embedding', 'subject_id', 'fold', 'accuracy_zscore', 'raw_accuracy', 'mantel_z', 
		'mantel_rho', 'accuracy_himalaya'])

	for X, embd in zip([tph_data, data], [1, 0]):
		r, p, z = run_rsa(X, target_coords)
		accuracies, zscored_accuracies = run_cv_linreg_with_null(X, target_coords, run_labels)
		ridge_accuracies = run_ridge_cv_himalaya(X, target_coords, run_labels)
		for i in range(4):
			df.loc[len(df)] = {'embedding':embd, 'subject_id':subject_id, 'fold':i, 'accuracy_zscore':zscored_accuracies[i], 
			'raw_accuracy':accuracies[i], 'accuracy_himalaya':ridge_accuracies[i], 'mantel_z':z, 'mantel_rho':r}
	df.to_csv(f'{DATA_PATH}/{subject_id}_joystick_results_final.csv')
	print(f'Finished {subject_id}')
	return df


if __name__ == "__main__":
	joblist = []
	joblist = [delayed(drive_joystick_analyses)(S) for S in SUB_IDS]
	with Parallel(n_jobs=16) as parallel:
		results_list = parallel(joblist)
	results_list=pd.concat(results).reset_index(drop=True)
	results_list.to_csv(f'{INTERMEDIATE_RESULTS_PATH}/joystick_decoding_results.csv')


