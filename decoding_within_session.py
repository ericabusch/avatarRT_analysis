# decoding_within_session_results.py
# Plots MSE of decoders trained and tested within the same session type,
# as in revision.ipynb cell 53 (make_barplot_points on df_cong_avg).
#
# Source file: results/results_public/decoding_results_aug6_cross_session_run_cross_validation.csv
# Output plot: results/plots/decoders_within_session.pdf

import numpy as np
import pandas as pd
import os
import analysis_helpers as helper
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True
import matplotlib.pyplot as plt
import seaborn as sns
from plotting_functions import make_barplot_points_precomputed
from config import *
from sklearn.model_selection import PredefinedSplit, KFold
from sklearn.metrics import mean_squared_error
from himalaya.ridge import RidgeCV, Ridge
from sklearn.linear_model import LinearRegression



RESULTS_PUBLIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'results_public')
PLOTS_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'plots')
DECODING_FN    = os.path.join(RESULTS_PUBLIC, 'decoding_results_aug6_cross_session_run_cross_validation.csv')

os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_context(context_params)

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

def himalaya_regression(X_train, X_test, y_train, y_test,inner_cv=True):    
    if inner_cv:
        inner_CV = KFold(5)
        reg=RidgeCV(alphas=ALPHAS, cv=inner_CV).fit(X_train, y_train)
    else: 
        reg=Ridge(alpha=0.1).fit(X_train, y_train)
    yhat=reg.predict(X_test)
    mse=mean_squared_error(y_test, yhat)
    return mse

def run_cv_himalaya(X, target_mat, cv_labels, inner_cv=False):
    ps=PredefinedSplit(cv_labels)
    results=[]
    for i, (train_index, test_index) in enumerate(ps.split()):
        results.append(himalaya_regression(X[train_index], X[test_index], target_mat[train_index], target_mat[test_index], inner_cv=inner_cv))
    return results

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


def main():
    if not os.path.exists(DECODING_FN):
        print(f'File not found: {DECODING_FN}')
        return

    df = pd.read_csv(DECODING_FN, index_col=0)
    df = df[~df['subject_id'].isin(['avatarRT_sub_09', 'avatarRT_sub_20'])]

    # Within-session rows: trained and tested on the same session type
    df_cong = df[df['congruent'] == True].copy()

    # Average across folds and runs → one MSE value per subject × session type
    df_cong_avg = (df_cong
                   .groupby(['subject_id', 'test_session_type'], as_index=False)['mse']
                   .mean()
                   .rename(columns={'test_session_type': 'session_type'}))

    print(f'Within-session data: {df_cong_avg.shape[0]} rows '
          f'({df_cong_avg["subject_id"].nunique()} subjects × 3 session types)')
    print(df_cong_avg.groupby('session_type')['mse'].describe().round(3))

    # Bootstrap CI, permutation tests (via make_barplot_points below), and Cohen's d
    d_wide = df_cong_avg.pivot(index='subject_id', columns='session_type', values='mse').dropna()
    v_im = d_wide['IM'].values
    v_wm = d_wide['WMP'].values
    v_om = d_wide['OMP'].values

    m_im, lo_im, hi_im, _   = helper.bootstrap_ci(v_im, n_boot=10000, verbose=0)
    m_wm, lo_wm, hi_wm, _   = helper.bootstrap_ci(v_wm, n_boot=10000, verbose=0)
    m_om, lo_om, hi_om, _   = helper.bootstrap_ci(v_om, n_boot=10000, verbose=0)
    _, p_im_0,  _ = helper.permutation_test(np.array([v_im, np.zeros(len(v_im))]),  10000, alternative='greater')
    _, p_wm_0,  _ = helper.permutation_test(np.array([v_wm, np.zeros(len(v_wm))]),  10000, alternative='greater')
    _, p_om_0,  _ = helper.permutation_test(np.array([v_om, np.zeros(len(v_om))]),  10000, alternative='greater')
    _, p_im_wm, _ = helper.permutation_test(np.array([v_im, v_wm]), 10000, alternative='two-sided')
    _, p_im_om, _ = helper.permutation_test(np.array([v_im, v_om]), 10000, alternative='two-sided')
    _, p_wm_om, _ = helper.permutation_test(np.array([v_wm, v_om]), 10000, alternative='two-sided')
    m_im_wm,  lo_im_wm,  hi_im_wm,  _ = helper.bootstrap_ci(v_im - v_wm, n_boot=10000, verbose=0)
    m_im_om,  lo_im_om,  hi_im_om,  _ = helper.bootstrap_ci(v_im - v_om, n_boot=10000, verbose=0)
    m_wm_om,  lo_wm_om,  hi_wm_om,  _ = helper.bootstrap_ci(v_wm - v_om, n_boot=10000, verbose=0)
    d_im_wm = helper.cohens_d_paired(v_im - v_wm, verbose=0)
    d_im_om = helper.cohens_d_paired(v_im - v_om, verbose=0)
    d_wm_om = helper.cohens_d_paired(v_wm - v_om, verbose=0)
    d_im_0 = helper.cohens_d_paired(v_im, verbose=0)
    d_wm_0 = helper.cohens_d_paired(v_wm, verbose=0)
    d_om_0 = helper.cohens_d_paired(v_om, verbose=0)

    print('\n--- Within-session MSE: bootstrap CIs, permutation tests, Cohen\'s d ---')
    print(f'IM:   mean={m_im:.4f}  95%CI=[{lo_im:.4f},{hi_im:.4f}]  p_vs_0={p_im_0:.4f}  d={d_im_0:.4f}')
    print(f'WMP:  mean={m_wm:.4f}  95%CI=[{lo_wm:.4f},{hi_wm:.4f}]  p_vs_0={p_wm_0:.4f}  d={d_wm_0:.4f}')
    print(f'OMP:  mean={m_om:.4f}  95%CI=[{lo_om:.4f},{hi_om:.4f}]  p_vs_0={p_om_0:.4f}  d={d_om_0:.4f}')
    print(f'IM  vs WMP:  mean={m_im_wm:.4f}  p={p_im_wm:.4f}  95%CI=[{lo_im_wm:.4f},{hi_im_wm:.4f}]  d={d_im_wm:.4f}')
    print(f'IM  vs OMP:  mean={m_im_om:.4f}  p={p_im_om:.4f}  95%CI=[{lo_im_om:.4f},{hi_im_om:.4f}]  d={d_im_om:.4f}')
    print(f'WMP vs OMP:  mean={m_wm_om:.4f}  p={p_wm_om:.4f}  95%CI=[{lo_wm_om:.4f},{hi_wm_om:.4f}]  d={d_wm_om:.4f}')

    all_stats = []
    for label, vals, p, cohd, ci_lo, ci_hi, alt in [
        ('IM',  v_im, p_im_0, d_im_0, lo_im, hi_im, 'greater'),
        ('WMP', v_wm, p_wm_0, d_wm_0, lo_wm, hi_wm, 'greater'),
        ('OMP', v_om, p_om_0, d_om_0, lo_om, hi_om, 'greater'),
    ]:
        all_stats.append({
            'comparison': f'MSE: {label} vs 0',
            'test': f'permutation_test (n_iter=10000, alternative={alt})',
            'group1': label, 'group2': '0 (null)',
            'n1': len(vals), 'n2': np.nan,
            'mean1': np.mean(vals), 'mean2': 0,
            'p_value': p, 'ci_lower': ci_lo, 'ci_upper': ci_hi,
            'cohens_d': cohd,
        })
    for label, g1, g2, g1v, g2v, diff_vals, p, cohd, ci_lo, ci_hi, alt in [
        ('IM vs WMP', 'IM', 'WMP', v_im, v_wm, v_im - v_wm, p_im_wm, d_im_wm, lo_im_wm, hi_im_wm, 'two-sided'),
        ('IM vs OMP', 'IM', 'OMP', v_im, v_om, v_im - v_om, p_im_om, d_im_om, lo_im_om, hi_im_om, 'two-sided'),
        ('WMP vs OMP','WMP','OMP', v_wm, v_om, v_wm - v_om, p_wm_om, d_wm_om, lo_wm_om, hi_wm_om, 'two-sided'),
    ]:
        all_stats.append({
            'comparison': f'MSE: {label}',
            'test': f'permutation_test (n_iter=10000, alternative={alt})',
            'group1': g1, 'group2': g2,
            'n1': len(g1v), 'n2': len(g2v),
            'mean1': np.mean(g1v), 'mean2': np.mean(g2v),
            'p_value': p, 'ci_lower': ci_lo, 'ci_upper': ci_hi,
            'cohens_d': cohd,
        })
    stats_df = pd.DataFrame(all_stats)
    stats_df['significant_0.05'] = stats_df['p_value'] < 0.05
    stats_fn = os.path.join(RESULTS_PUBLIC, 'decoding_within_session_stats.csv')
    stats_df.to_csv(stats_fn, index=False)
    print(f'Saved statistics to {stats_fn}')

    out_fn = os.path.join(PLOTS_DIR, 'decoders_within_session.pdf')
    fig, ax = make_barplot_points_precomputed(
        df_cong_avg, 'mse', 'session_type',
        pvals_vs_0=[p_im_0, p_wm_0, p_om_0],
        pvals_pairwise=[p_im_wm, p_im_om, p_wm_om],
        exclude_subs=[9, 20],
        ylim=[0, 1.5],
        plus_bot=0.2, plus_top=0.35,
        ylabel='MSE', xlabel='Session type',
    )
    plt.savefig(out_fn, transparent=True, bbox_inches='tight', format='pdf')
    plt.close()
    print(f'Saved plot: {out_fn}')


if __name__ == '__main__':
    main()
