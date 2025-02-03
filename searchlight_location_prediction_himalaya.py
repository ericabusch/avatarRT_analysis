import numpy as np
import pandas as pd
import os, sys, glob, shutil 
sys.path.insert(0,'..')
sys.path.insert(0,'../..')
import argparse
from mpi4py import MPI
import pickle
import matplotlib.pyplot as plt
import avatarRT_utils as utils
import inspect, subprocess
from numpy import linalg
from scipy import stats
import nibabel as nib
from scipy.stats import spearmanr, pearsonr,zscore
from scipy.stats import f_oneway, spearmanr
from nilearn.maskers import NiftiMasker
from nilearn import plotting  
from nibabel import Nifti1Image
from nilearn import image
from himalaya.kernel_ridge import KernelRidgeCV
from brainiak.searchlight.searchlight import Searchlight
from nilearn.datasets import load_mni152_template
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import analysis_helpers as helper
from analysis_helpers import shift_timing, load_location_labels, list_by_trial
from config import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import PredefinedSplit


def load_brain_data(subject_id, session_id, run, mask_file=None, normalize=True, subsample_mask=False):
    nii = utils.load_vol_data(subject_id, session_id, run, space='standard', asarray=False)
    bold_data = nii.get_fdata()
    bold_data = np.nan_to_num(utils.normalize(bold_data, axis=-1))
    dims = nii.header.get_zooms()
    affine=nii.affine
    if mask_file == None:
        brain_mask,_,_=get_mask(subsample_mask=subsample_mask)
    else:
        brain_mask=nib.load(mask_file).get_fdata()

    return bold_data, brain_mask, affine, dims

def get_mask(subsample_mask=False):
    nii=nib.load(INTERSECT_MASK)
    M = nii.get_fdata()
    if subsample_mask:
        Ms = np.zeros_like(M)
        coords=np.array(np.where(M==1))
        subsamp=coords[:,::2]
        Ms[subsamp[0],subsamp[1],subsamp[2]]=1
        M=Ms
    return M, nii.affine, nii.header.get_zooms()


def get_num_trials(subject_id, session_id, run):
    reg_df = pd.read_csv(f'{utils.data_path}/{subject_id}/regressors/{subject_id}_{session_id}_run_{run:02d}_timeseries_regressors.csv', index_col=0)
    trial_numbers = reg_df['trial'].values
    return len(np.unique(trial_numbers))-1

def prediction_kernel(data, sl_mask, radius, bcvars):
    # data will be 4d - one entry for each trial
    n_trials = len(data)
    prediction_error = np.zeros((n_trials,))
    best_alpha=np.zeros((n_trials,2))
    if np.sum(sl_mask) < radius*3*2: 
        print(f'TOO SMALL')
        return np.nan
    xcoords = bcvars[0][0][0]
    zcoords = bcvars[0][0][1]
    num_voxels_in_sl = sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2]

    data_all = np.empty((0, num_voxels_in_sl))
    for ti in range(n_trials):
        data_ti = np.nan_to_num(data[ti].reshape(num_voxels_in_sl, data[ti].shape[-1]).T)
        data_all = np.concatenate((data_all, data_ti))

    trial_labels = np.concatenate([[k for i in range(data[k].shape[-1])] for k in range(len(data))])
    
    x_all = np.concatenate(xcoords)
    z_all = np.concatenate(zcoords)
    assert len(trial_labels)==len(x_all)==len(data_all)
    assert data_all.shape[1] == num_voxels_in_sl

    outer_split = PredefinedSplit(trial_labels)
    for t, (train_trials_idx, test_trial_idx) in enumerate(outer_split.split()):
        data_test = data_all[test_trial_idx]
        data_train = data_all[train_trials_idx]
        xtrain,ztrain=x_all[train_trials_idx],z_all[train_trials_idx]
        xtest, ztest = x_all[test_trial_idx], z_all[test_trial_idx]
        TEST_TARGETS=np.array((xtest,ztest)).T
        TRAIN_TARGETS=np.array((xtrain,ztrain)).T
        inner_trial_labels = trial_labels[trial_labels!=t]

        if (np.sum(xtrain!=xtrain) > 0) or (np.sum(ztrain!=ztrain) > 0) or (data_test.shape[0]==0):
            # if data_test.shape[0] == 0: print(f"test data {t} has no samples; train data has {data_train.shape}")
            # else: print('training labels have nans') 
            prediction_error[t]=np.nan 
            continue

        # Run ridge regression with CV
        inner_CV = PredefinedSplit(inner_trial_labels)
        model = KernelRidgeCV(alphas=ALPHAS, cv=inner_CV, kernel='linear').fit(data_train, TRAIN_TARGETS)
        best_alpha[t]=model.best_alphas_
        predicted_targets = model.predict(data_test)
        prediction_error[t] = np.mean(linalg.norm(predicted_targets-TEST_TARGETS, axis=1))
    return prediction_error, best_alpha

def parse_sl_results(result_vol, mask_coords):
    INIT_MTX = lambda a,b,c : np.full((a, b, c), np.nan)
    result_vec = result_vol[mask_coords[0], mask_coords[1], mask_coords[2]]
    n_coords = len(result_vec)
    n_trials = result_vec[0][0].shape[0]
    n_targets = 2
    SELECTED_ALPHAS = INIT_MTX(n_coords,n_trials,n_targets)
    PREDICTIONS = INIT_MTX(n_coords,n_trials,1).squeeze()

    # handle predictions
    no_vals=0
    for v in range(n_coords):
        try:
            r=result_vec[v][0]
            PREDICTIONS[v]=np.where(r==0, np.nan, r)
        except:
            no_vals+=1
            continue
    print(f'{no_vals} voxels had no prediction values')

    # handle alphas; doesnt matter to have 0s, not analyzing
    no_vals=0
    for v in range(n_coords):
        try:
            SELECTED_ALPHAS[v,:]=result_vec[v][1:][0]
        except:
            no_vals+=1
            continue
    print("returning from parsing results")
    return PREDICTIONS, SELECTED_ALPHAS

def plot_volume(result_vector, mask_nii, title, filename_base):
    M = mask_nii.get_fdata()
    affine=mask_nii.affine
    dimsize=mask_nii.header.get_zooms()
    result_vol = np.zeros_like(M)
    coords=np.where(M==1)
    result_vol[coords[0],coords[1],coords[2]]=result_vector
    result_vol = result_vol.astype('double')
    result_vol = np.nan_to_num(result_vol)
    nii = Nifti1Image(result_vol, affine=affine)
    nii.header.set_zooms(dimsize[:result_vol.ndim])
    nib.save(nii, filename_base+'.nii.gz')
    if p.verbose: print(f'saved to {filename_base}')
    plotting.plot_stat_map(nii, bg_img=ANAT_FILE, output_file=filename_base+'.png', colorbar=True, title=title, cmap='viridis')
    if p.verbose: print(f'saved plot of title {title}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sub','--subject_id',type=str)
    parser.add_argument('-ses','--session_number', type=int)
    parser.add_argument('-rad','--sl_rad', type=int, default=5)
    parser.add_argument('-t','--test', type=int, default=0)
    parser.add_argument('-v','--verbose', type=int, default=1)
    parser.add_argument('-o', '--overwrite', type=int, default=1)
    parser.add_argument('-p', '--plot', type=int, default=0)
    p = parser.parse_args()

    COMM = MPI.COMM_WORLD
    RANK = COMM.rank
    SIZE = COMM.size
    max_blk_edge = 10
    pool_size = None
    NORMALIZE=1
    SUBSAMP_MASK=1

    subject_id=p.subject_id
    session_id = f'ses_{p.session_number:02d}'
    ALPHAS = 10.**np.arange(-2, 20, 1)
    #run_numbers = [1,2,3,4]
    #if p.session_number > 2: run_numbers=[2,3,4]
    run_numbers = [2,4]
    if subject_id == f'avatarRT_sub_06' and p.session_number == 3: 
        run_numbers = [1,2,4]
    elif p.session_number == 2:
        run_numbers = [1,2,4]
    sl = Searchlight(sl_rad=p.sl_rad, max_blk_edge=max_blk_edge,min_active_voxels_proportion=0.8)
    subjPath=f'{utils.project_path}/experiment/subjects/{subject_id}'
    os.makedirs(subjPath+'/results/searchlight',exist_ok=True)
    FSLDIR='/gpfs/milgram/apps/hpc.rhel7/software/FSL/6.0.5-centos7_64'
    MASK_FILE=  f'{FSLDIR}/data/linearMNI/MNI152lin_T1_2mm_brain_mask.nii.gz'
    ANAT_FILE = f'{FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz'
    INTERSECT_MASK=f'{subjPath}/masks/{subject_id}_{session_id}_intersect_mask.nii.gz'
    if p.test==1:
        if p.verbose: print('using test mask')
        INTERSECT_MASK=f'/gpfs/milgram/project/turk-browne/users/elb77/BCI/rt-cloud/projects/avatarRT/offline_analyses/TEST_MASK.nii.gz'
    FAILED_STACK=0
    run_mean_results = []
    mask,affine,dimsize=get_mask()
    coords = np.where(mask==1)
    print(f'mask has {np.sum(mask)} voxels')
    SCRATCH_OUTPATH=f'{SCRATCH_PATH}/sl_analyses_himalaya'
    SUBJ_OUTPATH=f'{subjPath}/results/sl_analyses_himalaya/'
    os.makedirs(SUBJ_OUTPATH, exist_ok=True)
    os.makedirs(SCRATCH_OUTPATH, exist_ok=True)
    mask,affine,dimsize=get_mask(subsample_mask=SUBSAMP_MASK)
    for run in run_numbers:
        data,masks,affines,dimsizes,bcvars = [],[],[],[],[]
        perturb_type = utils.get_perturbation_info(subject_id, session_id, run, return_component=False)
        nm='ridge_prediction_euclid_dist'
        root=f'{subject_id}_{session_id}_{perturb_type}_run_{run:02d}_{p.sl_rad}_searchlight_{nm}'
        if p.test: root+='_TEST'
        outname = f'{SUBJ_OUTPATH}/{root}'
        f0 = f'{SCRATCH_OUTPATH}/{root}_mean.npy'
        N = get_num_trials(subject_id, session_id, run)
        if os.path.exists(f0):
            run_mean_results.append(np.load(f0))
            if p.verbose: print(f'LOADED BACK IN RESULTS from {f0}; CONTINUING TO THE NEXT RUN!\n\n')
            continue
        all_results_dump = f'{SCRATCH_OUTPATH}/{root}_all_sl_results_raw.pkl'
        
        elif os.path.exists(all_results_dump):
            # load back in these results
            if RANK==0:
                if p.verbose: print(f'Loading back in pickled data from {all_results_dump}!\n\n')
                with open(all_results_dump,'rb') as f:
                    all_sl_results=pickle.load(f)
                try:
                    PREDICTIONS,SELECTED_ALPHAS=parse_sl_results(all_sl_results, coords)
                    np.save(alphas_fn, SELECTED_ALPHAS)
                    np.save(f'{SCRATCH_OUTPATH}/{root}_result_vec.npy', PREDICTIONS)
                    if p.verbose: print(f'saved alphas and predictions')
                    # average the results for each measure
                    MU=np.nanmean(PREDICTIONS,axis=1)
                    STD=np.nanstd(PREDICTIONS,axis=1)
                    np.save(f'{SCRATCH_OUTPATH}/{root}_mean.npy', MU)
                    np.save(f'{SCRATCH_OUTPATH}/{root}_std.npy', STD)
                    run_mean_results.append(MU)
                    if p.verbose: print(f'got run mean')
                    if p.plot: 
                        mask_nii=nib.load(INTERSECT_MASK)
                        title = f'{subject_id} {session_id} run_{run:02d} {perturb_type}'
                        outfn_base = f'{SUBJ_OUTPATH}/{root}_mean'
                        plot_volume(MU, mask_nii, title, outfn_base)
                except:
                    print(f'still failed after loding pickled data; continue')
                    continue

        else:
            if RANK == 0:
                # only load data on rank 0
                bold_data, mask, affine, dimsize = load_brain_data(subject_id, session_id, run, mask_file=INTERSECT_MASK, subsample_mask=SUBSAMP_MASK)
                trial, xvals, zvals = load_location_labels(subject_id, session_id, run, bold_data.shape[-1])
                xvals, zvals = np.nan_to_num(xvals), np.nan_to_num(zvals)
                if p.verbose: print(f'loaded data of shape {bold_data.shape} for {subject_id} {session_id} {perturb_type} run {run}, n_trials={len(np.unique(trial))-1}')
                # now turn it into trials
                data = list_by_trial(bold_data, trial, normalize=False)
                x_list = list_by_trial(xvals, trial, normalize=NORMALIZE)
                z_list = list_by_trial(zvals, trial, normalize=NORMALIZE)
                affines.append(affine)
                dimsizes.append(dimsize)
                bcvars.append([[x_list, z_list]])
                masks.append(mask)
            else:
                data = [None for i in range(N)]
            
            sl = Searchlight(sl_rad=p.sl_rad, max_blk_edge=max_blk_edge)
            sl.distribute(data, mask)
            sl.broadcast(bcvars)

            if p.verbose: print(f"Begin Searchlight in rank {RANK}")
            all_sl_results = sl.run_searchlight(prediction_kernel, pool_size=pool_size)
            if p.verbose: print(f"End Searchlight in rank {RANK}; results of shape {np.shape(all_sl_results)})")
            
            alphas_fn = f'{SCRATCH_OUTPATH}/{root}_all_sl_trial_alphas.npy'
            # invert if on rank 0
            if RANK == 0:
                # save the all sl results
                with open(all_results_dump,'wb') as f:
                    pickle.dump(all_sl_results, f)

                print(f'saved to: {all_results_dump}')
                # separate out the alphas from the actual results
                result_vec = all_sl_results[coords[0], coords[1], coords[2]]
                print(f'result vec: {result_vec.shape}')
                try:
                    PREDICTIONS,SELECTED_ALPHAS=parse_sl_results(all_sl_results, coords)
                    np.save(alphas_fn, SELECTED_ALPHAS)
                    np.save(f'{SCRATCH_OUTPATH}/{root}_result_vec.npy', PREDICTIONS)
                    if p.verbose: print(f'saved alphas and predictions')
                    # average the results for each measure
                    MU=np.nanmean(PREDICTIONS,axis=1)
                    STD=np.nanstd(PREDICTIONS,axis=1)
                    np.save(f'{SCRATCH_OUTPATH}/{root}_mean.npy', MU)
                    np.save(f'{SCRATCH_OUTPATH}/{root}_std.npy', STD)
                    run_mean_results.append(MU)
                    if p.verbose: print(f'got run mean')
                    if p.plot: 
                        mask_nii=nib.load(INTERSECT_MASK)
                        title = f'{subject_id} {session_id} run_{run:02d} {perturb_type}'
                        outfn_base = f'{SUBJ_OUTPATH}/{root}_mean'
                        plot_volume(MU, mask_nii, title, outfn_base)

                except:
                    FAILED_STACK+=1
                    print(f'Something failed in the try; probably just couldnt stack results, continuing to next run')
                    continue
                
    print(f'Handled all runs; run mean list of len {len(run_mean_results)}')
    if RANK != 0: sys.exit()
    if FAILED_STACK > 0:
        print(f'Failed stacking some result; reload from pickles')
        sys.exit()
    # now, if on rank 0, subtract the first and last volumes within each metric
    if p.verbose: print(f'Results of shape: ',run_mean_results[0].shape, run_mean_results[1].shape)
    nm='ridge_prediction_euclid_dist_difference'
    root=f'{subject_id}_{session_id}_{perturb_type}_{p.sl_rad}_searchlight_{nm}'
    # add them and normalize
    r0 = run_mean_results[0]
    r1 = run_mean_results[-1]
    np.save(f'{SUBJ_OUTPATH}/{root}_raw_difference.npy', r0 - r1)
    norm_dif = (r0 - r1) / (r0 + r1)
    np.save(f'{SUBJ_OUTPATH}/{root}_normalized_difference.npy', norm_dif)
    title=root
    mask_nii=nib.load(INTERSECT_MASK)
    filename_base=f'{SUBJ_OUTPATH}/{root}_normalized_difference'
    plot_volume(norm_dif, mask_nii, title, filename_base)
    



