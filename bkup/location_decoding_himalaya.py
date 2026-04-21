import numpy as np
import pandas as pd
import os,sys,glob,inspect,argparse
sys.path.insert(0,'../../')
from TPHATE.tphate import tphate
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr, zscore, percentileofscore
from sklearn.model_selection import PredefinedSplit, KFold
from sklearn.metrics import mean_squared_error
from himalaya.ridge import RidgeCV, Ridge
import analysis_helpers as helper
import nibabel as nib
import mantel
from config import *
from joblib import Parallel, delayed
from sklearn.decomposition import FactorAnalysis, PCA
ALPHAS=10.**np.arange(-5, 5, 1)

def run_rsa(X, target_mat):
    Ymat=1-pdist(target_mat,'euclidean') # similarity
    Xmat=1-pdist(X, 'correlation') # correlation, not corr - dist
    mantel_result = mantel.test(Xmat, Ymat, perms=NPERM, method='spearman', tail='upper',ignore_nans=True)
    return mantel_result.r, mantel_result.p, mantel_result.z

def himalaya_regression(X_train, X_test, y_train, y_test,inner_cv=True):    
    if inner_cv:
        inner_CV = KFold(5)
        reg=RidgeCV(alphas=ALPHAS, cv=inner_CV).fit(X_train, y_train)
    else: 
        reg=Ridge(alpha=0.1).fit(X_train, y_train)
    yhat=reg.predict(X_test)
    mse=mean_squared_error(y_test, yhat)
    return mse

def run_cv_linreg(X, target_mat, cv_labels, inner_cv=False):
    ps=PredefinedSplit(cv_labels)
    results=[]
    for i, (train_index, test_index) in enumerate(ps.split()):
        results.append(linear_regression2d(X[train_index], X[test_index], target_mat[train_index], target_mat[test_index]))
    return results

def linreg_prediction(X_train, X_test, y_train):
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_hat = linreg.predict(X_test)
    return y_hat

def linear_regression2d(X_train, X_test, y_train, y_test, inner_cv=False):
    yhat_xcoord = linreg_prediction(X_train, X_test, y_train[:, 0])
    yhat_zcoord = linreg_prediction(X_train, X_test, y_train[:, 1])
    yhat = np.array((yhat_xcoord, yhat_zcoord)).T
    return mean_squared_error(y_test, yhat)

def get_tphate_embedding(subject_id, voxel_data, ndim=20, rerun=True): 
    temp_fn = f'{SCRATCH_PATH}/joystick_analyses/{subject_id}_{ndim}d_TPHATE_embedding.npy'
    if os.path.exists(temp_fn) and not rerun:
        tph_dst=np.load(temp_fn)
        return tph_dst
    tph_dst = tphate.TPHATE(verbose=0, n_components=ndim, t=5).fit_transform(voxel_data)
    np.save(temp_fn, tph_dst)
    return tph_dst

def get_pca_embedding(subject_id, voxel_data, ndim=20, rerun=True): 
    temp_fn = f'{SCRATCH_PATH}/joystick_analyses/{subject_id}_{ndim}d_pcs.npy'
    if os.path.exists(temp_fn) and not rerun:
        dst=np.load(temp_fn)
        return dst
    dst = PCA(n_components=ndim).fit_transform(voxel_data)
    np.save(temp_fn, dst)
    return dst

def load_raw_joystick_labels(subject_id, run, shift_by=2, concat=True):
    mask_arr = nib.load(f'{DATA_PATH}/{subject_id}/reference/mask.nii.gz').get_fdata()
    _, ti, xi, zi = helper.load_all_joystick_data(subject_id, run, mask_arr, shift_by)
    # raw targets:
    xt=xi[ti>=0]
    zt=zi[ti>=0]
    raw_targets = np.array((xt, zt)).T
    return raw_targets

def load_joystick_data_package(subject_id, concat=True, shift_by=2, get_raw_loc=True):
    mask_arr = nib.load(f'{DATA_PATH}/{subject_id}/reference/mask.nii.gz').get_fdata()
    data, trial_labels, x_vals, z_vals, run_labels =  [], [], [], [], []
    raw_locations = []
    for run in range(1,5):
        ds, ti, xi, zi = helper.load_all_joystick_data(subject_id, run, mask_arr, shift_by)
        # normalizes the labels within trial
        x_normed = helper.normalize_within_trial(xi, ti)
        z_normed = helper.normalize_within_trial(zi, ti)
        if get_raw_loc:
            xt=xi[ti>=0]
            zt=zi[ti>=0]
            raw_locations.append(np.array((xt, zt)).T)
        
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
        raw_locations=np.concatenate(raw_locations,axis=0)
    if VERBOSE: print(len(data), len(xs), len(run_labels),raw_locations.shape)
    
    return np.nan_to_num(data), xs, zs, run_labels, trial_labels, raw_locations

def load_RT_data_package(subject_id, session_id, concat=True, shift_by=2, data_type='projected_data'):
    data, trial_labels, x_vals, z_vals, run_labels =  [], [], [], [], []
    raw_locations = []
    if session_id == 'ses_02':
        runs = [1,2,3,4]
    else:
        runs = [1,2,3,4]
        
    for run in runs:
        
        # load in desired brain data for this run
        ds = helper.get_realtime_outdata(subject_id, session_id, run, data_type=data_type)
        
        # load in trials, xlabels, zlabels, shifted, for this data
        ti, xi, zi = helper.load_location_labels(subject_id, session_id, run, ds.shape[0], shift_by=shift_by)
        print(ds.shape, xi.shape,ti.shape)
        
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
    
    # drop out nans
    data_nans = np.where(data != data)[0] # will exclude the whole row
    xnans = np.where(xs != xs)[0] 
    znans = np.where(zs != zs)[0]
    idx_to_drop = np.unique(np.concatenate([data_nans, xnans, znans]))
    if len(idx_to_drop) > 0:
        idx_to_include = np.setdiff1d(np.arange(len(data)), idx_to_drop)
        data = data[idx_to_include]
        xs = xs[idx_to_include]
        zs = zs[idx_to_include]
        run_labels = run_labels[idx_to_include]
        trial_labels = trial_labels[idx_to_include]
    if VERBOSE: print(len(data), len(xs), len(run_labels), 'dropped: ', len(idx_to_drop))
    return np.nan_to_num(data), xs, zs, run_labels, trial_labels

def run_cv_himalaya(X, target_mat, cv_labels, inner_cv=False):
    ps=PredefinedSplit(cv_labels)
    results=[]
    for i, (train_index, test_index) in enumerate(ps.split()):
        results.append(himalaya_regression(X[train_index], X[test_index], target_mat[train_index], target_mat[test_index], inner_cv=inner_cv))
    return results

def run_figure1_analysis(subject_id, cross_validation, regression_function='himalaya', n_dim=20, hyperparam_op=False):
    data, x_coords, z_coords, run_labels, trial_labels, raw_locations = load_joystick_data_package(subject_id)
    tphate_embedding = get_tphate_embedding(subject_id, data, ndim=n_dim)
    pca_embedding = get_pca_embedding(subject_id, data, ndim=n_dim)
    target_coords = np.nan_to_num(np.array((x_coords, z_coords)).T)
    if cross_validation == 'run':
        cv_labels=run_labels
    else:
        cv_labels=trial_labels
    
    if regression_function == 'himalaya':
        CV_FUNC=run_cv_himalaya
    else:
        CV_FUNC=run_cv_linreg
    
    
    dfs = []
    for name,x in zip(['voxel','tphate','pca'], [data, tphate_embedding, pca_embedding]):
        accs = CV_FUNC(x, target_coords, cv_labels, hyperparam_op)
        r,p,z= run_rsa(x, raw_locations)
        v=len(accs)+3
        temp = pd.DataFrame({'subject_id':[subject_id]*v,
                'embedding_dim':[n_dim] * v, 
                'embedding_type':name,
                'cv':cross_validation,
                'fold':[i for i in range(len(accs))]+['na','na','na'],
                'metric':[regression_function for i in range(len(accs))]+['r','p','z'],
                'score':accs + [r, p, z]
                })
        dfs.append(temp)
    return pd.concat(dfs)

def run_supp_dimensionality_analysis(subject_id, cross_validation, dim2test=[2,3,5,10,15,20], hyperparam_op=False, regression_function='himalaya'):
    data, x_coords, z_coords, run_labels, trial_labels,raw_locations = load_joystick_data_package(subject_id)
    target_coords = np.nan_to_num(np.array((x_coords, z_coords)).T)
    
    if cross_validation == 'run':
        cv_labels=run_labels
    else:
        cv_labels=trial_labels
    
    if regression_function == 'himalaya':
        CV_FUNC=run_cv_himalaya
    else:
        CV_FUNC=run_cv_linreg
    
    dfs = []
    for ndim in dim2test:
        tphate_embedding = get_tphate_embedding(subject_id, data, ndim=ndim)
        pca_embedding = get_pca_embedding(subject_id, data, ndim=ndim)
        fa_embedding = get_factor_analysis_embedding(subject_id, data, ndim=ndim)
        for name,x in zip(['tphate','pca','fa'], [tphate_embedding, pca_embedding, fa_embedding]):
            accs = CV_FUNC(x, target_coords, cv_labels, hyperparam_op)
            r,p,z= run_rsa(x,raw_locations)
            v=len(accs)+3
            temp = pd.DataFrame({'subject_id':[subject_id]*v,
                    'embedding_dim':[ndim] * v, 
                    'embedding_type':name,
                    'cv':cross_validation,
                    'fold':[i for i in range(len(accs))]+['na','na','na'],
                    'metric':[regression_function for i in range(len(accs))]+['r','p','z'],
                    'score':accs + [r, p, z]
                    })
            dfs.append(temp)
    return pd.concat(dfs)

def run_cross_session_decoders(subject_id, cross_validation, regression_function='himalaya'):
    cumulative_info = helper.load_info_file()
    subject_info = cumulative_info[cumulative_info['subject_ID'] == subject_id ]
    im_session=f'ses_0'+str(int(subject_info['im_session'].item())) 
    wmp_session=f'ses_0'+str(int(subject_info['wmp_session'].item())) 
    omp_session=f'ses_0'+str(int(subject_info['omp_session'].item()))
    
    
    data_IM, x_IM, z_IM, run_label_IM, trial_labels_IM = load_RT_data_package(subject_id, im_session, data_type='projected_data')
    targets_IM = np.array((x_IM, z_IM)).T
    
    data_WMP, x_WMP, z_WMP, run_label_WMP, trial_labels_WMP = load_RT_data_package(subject_id, wmp_session, data_type='projected_data')
    targets_WMP = np.array((x_WMP, z_WMP)).T
    
    data_OMP, x_OMP, z_OMP, run_label_OMP, trial_labels_OMP = load_RT_data_package(subject_id, omp_session, data_type='projected_data')
    targets_OMP = np.array((x_OMP, z_OMP)).T
    
    # first, train/test model with LORO CV on IM data, WM data, OMP data
    if regression_function == 'himalaya':
        CV_FUNC=run_cv_himalaya
        REG_FUNC=himalaya_regression
    else:
        CV_FUNC=run_cv_linreg
        REG_FUNC=linear_regression2d
    
    mse_IM = CV_FUNC(data_IM, targets_IM, run_label_IM, inner_cv=True)
    mse_WMP = CV_FUNC(data_WMP, targets_WMP, run_label_WMP, inner_cv=True)
    mse_OMP = CV_FUNC(data_OMP, targets_OMP, run_label_OMP, inner_cv=True)
    
    df = pd.DataFrame({'train_session_type':np.repeat(['IM','WMP','OMP'],4), 
                       'test_session_type':np.repeat(['IM','WMP','OMP'],4), 
                       'test_run':np.tile(np.arange(1,5),3), 
                       'fold':np.tile(np.arange(4),3), 
                       'mse':np.concatenate([mse_IM, mse_WMP, mse_OMP]), 
                       'congruent':[True]*12, 
                       'cv_type':['LORO']*12})
    
    # now, train model on IM data and test on runs of WM data
    for run in range(1,5):
        acc = REG_FUNC(data_IM, data_WMP[run_label_WMP==run], targets_IM, targets_WMP[run_label_WMP==run], inner_cv=True)
        df.loc[len(df)] = {'train_session_type':'IM',
                          'test_session_type':'WMP',
                          'test_run':run,
                          'fold':0,
                          'mse':acc,
                          'congruent':False,
                          'cv_type':'cross-session'}
        acc = REG_FUNC(data_IM, data_OMP[run_label_OMP==run], targets_IM, targets_OMP[run_label_OMP==run], inner_cv=True)
        df.loc[len(df)] = {'train_session_type':'IM',
                          'test_session_type':'OMP',
                          'test_run':run,
                          'fold':0,
                          'mse':acc,
                          'congruent':False,
                          'cv_type':'cross-session'}
    df['subject_id']=np.repeat(subject_id,len(df))
    df['regression_type']=np.repeat(regression_function,len(df))
    return df


if __name__ == "__main__":

    ################ Parse CL args #######################
    parser = argparse.ArgumentParser()
    parser.add_argument("-cv", "--cross_validation",type=str, help="run or trial", default='run')
    parser.add_argument("-v", '--verbose',type=int, default=1)
    parser.add_argument("-a", "--analysis",type=str, default="cross_session", help='which decoding?')
    parser.add_argument("-r", "--regression",type=str, default="himalaya", help='which regression model?')
    parser.add_argument("-d", "--n_dims",type=int, default=20)
    p = parser.parse_args()

    if p.analysis == 'figure1':
        func = run_figure1_analysis
    elif p.analysis == 'supp_dimensionality':
        func = run_supp_dimensionality_analysis
    elif p.analysis == 'cross_session':
        func = run_cross_session_decoders
    else:
        print('not yet implemented')
    print(f'running {p.analysis}, {p.regression}, cv={p.cross_validation}')
    joblist = [delayed(func)(S, p.cross_validation, regression_function=p.regression) for S in SUB_IDS]
    with Parallel(n_jobs=16) as parallel:
        results_list = parallel(joblist)
    results_list=pd.concat(results_list).reset_index(drop=True)
    results_list.to_csv(f'{INTERMEDIATE_RESULTS_PATH}/decoding_results_{p.analysis}_{p.cross_validation}_cross_validation.csv')