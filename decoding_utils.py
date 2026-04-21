# decoding_utils.py
# Shared utilities for location decoding analyses.
# Replaces location_decoding_himalaya.py; imported by run_joystick_analyses.py
# and run_RT_session_decoding.py.
#
# Contents
# --------
# Regression:   himalaya_regression, run_cv_himalaya, run_cv_linreg,
#               linreg_prediction, linear_regression2d
# Data loading: load_RT_data_package, load_joystick_data_package,
#               load_raw_joystick_labels
# Embeddings:   get_tphate_embedding, get_pca_embedding,
#               get_factor_analysis_embedding
# RSA:          run_rsa
# Session info: get_session_ids, select_cv_and_reg_funcs
# Analyses:     run_figure1_analysis, run_supp_dimensionality_analysis

import os
import numpy as np
import pandas as pd
import nibabel as nib
import mantel
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, PredefinedSplit
from himalaya.ridge import RidgeCV, Ridge
import tphate as tphate_module

import analysis_helpers as helper
from config import DATA_PATH, SCRATCH_PATH, ALPHAS, NPERM, VERBOSE

os.makedirs(os.path.join(SCRATCH_PATH, 'joystick_analyses'), exist_ok=True)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def himalaya_regression(X_train, X_test, y_train, y_test, inner_cv=True):
    if inner_cv:
        reg = RidgeCV(alphas=ALPHAS, cv=KFold(5)).fit(X_train, y_train)
    else:
        reg = Ridge(alpha=0.1).fit(X_train, y_train)
    return mean_squared_error(y_test, reg.predict(X_test))


def linreg_prediction(X_train, X_test, y_train):
    return LinearRegression().fit(X_train, y_train).predict(X_test)


def linear_regression2d(X_train, X_test, y_train, y_test, inner_cv=False):
    yhat = np.column_stack([
        linreg_prediction(X_train, X_test, y_train[:, 0]),
        linreg_prediction(X_train, X_test, y_train[:, 1]),
    ])
    return mean_squared_error(y_test, yhat)


def run_cv_himalaya(X, target_mat, cv_labels, inner_cv=True):
    ps = PredefinedSplit(cv_labels)
    return [
        himalaya_regression(X[tr], X[te], target_mat[tr], target_mat[te], inner_cv)
        for _, (tr, te) in enumerate(ps.split())
    ]


def run_cv_linreg(X, target_mat, cv_labels, inner_cv=False):
    ps = PredefinedSplit(cv_labels)
    return [
        linear_regression2d(X[tr], X[te], target_mat[tr], target_mat[te])
        for _, (tr, te) in enumerate(ps.split())
    ]


def select_cv_and_reg_funcs(regression_function):
    """Return (cv_func, reg_func) pair for the requested regression type."""
    if regression_function == 'himalaya':
        return run_cv_himalaya, himalaya_regression
    return run_cv_linreg, linear_regression2d


# ---------------------------------------------------------------------------
# Session metadata
# ---------------------------------------------------------------------------

def get_session_ids(subject_id):
    """Return (im_session, wmp_session, omp_session) string IDs for a subject."""
    info = helper.load_info_file()
    row  = info[info['subject_ID'] == subject_id]
    im   = f"ses_0{int(row['im_session'].item())}"
    wmp  = f"ses_0{int(row['wmp_session'].item())}"
    omp  = f"ses_0{int(row['omp_session'].item())}"
    print(f'{subject_id} sessions: IM={im}, WMP={wmp}, OMP={omp}')
    return im, wmp, omp


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_RT_data_package(subject_id, session_id, concat=True, shift_by=2,
                         data_type='projected_data'):
    data, trial_labels, x_vals, z_vals, run_labels = [], [], [], [], []

    for run in [1, 2, 3, 4]:
        ds       = helper.get_realtime_outdata(subject_id, session_id, run,
                                               data_type=data_type)
        ti, xi, zi = helper.load_location_labels(subject_id, session_id, run,
                                                  ds.shape[0], shift_by=shift_by)
        # if VERBOSE:
        #     print(ds.shape, xi.shape, ti.shape)
        x_normed = helper.normalize_within_trial(xi, ti)
        z_normed = helper.normalize_within_trial(zi, ti)
        # x_normed = xi
        # z_normed = zi
        in_trials = ti[ti >= 0]
        data.append(ds[ti >= 0])
        trial_labels.append(in_trials)
        x_vals.append(x_normed)
        z_vals.append(z_normed)
        run_labels.append(np.repeat(run, len(in_trials)))

    if concat:
        data         = np.concatenate(data, axis=0)
        xs           = np.concatenate(x_vals)
        zs           = np.concatenate(z_vals)
        run_labels   = np.concatenate(run_labels)
        trial_labels = np.concatenate(trial_labels)

        # drop NaN rows
        bad = np.unique(np.concatenate([
            np.where(data != data)[0],
            np.where(xs   != xs)[0],
            np.where(zs   != zs)[0],
        ]))
        if len(bad):
            keep = np.setdiff1d(np.arange(len(data)), bad)
            data, xs, zs = data[keep], xs[keep], zs[keep]
            run_labels, trial_labels = run_labels[keep], trial_labels[keep]
        # if VERBOSE:
        #     print(len(data), len(xs), len(run_labels), 'dropped:', len(bad))

    return np.nan_to_num(data), xs, zs, run_labels, trial_labels


def load_joystick_data_package(subject_id, concat=True, shift_by=2, get_raw_loc=True):
    mask_arr = nib.load(
        f'{DATA_PATH}/{subject_id}/reference/mask.nii.gz'
    ).get_fdata()
    data, trial_labels, x_vals, z_vals, run_labels, raw_locations = \
        [], [], [], [], [], []

    for run in range(1, 5):
        ds, ti, xi, zi = helper.load_all_joystick_data(subject_id, run,
                                                        mask_arr, shift_by)
        x_normed = helper.normalize_within_trial(xi, ti)
        z_normed = helper.normalize_within_trial(zi, ti)
        if get_raw_loc:
            raw_locations.append(np.array((xi[ti >= 0], zi[ti >= 0])).T)
        in_trials = ti[ti >= 0]
        data.append(ds[ti >= 0])
        trial_labels.append(in_trials)
        x_vals.append(x_normed)
        z_vals.append(z_normed)
        run_labels.append(np.repeat(run, len(in_trials)))

    if concat:
        data          = np.concatenate(data, axis=0)
        xs            = np.concatenate(x_vals)
        zs            = np.concatenate(z_vals)
        run_labels    = np.concatenate(run_labels)
        trial_labels  = np.concatenate(trial_labels)
        raw_locations = np.concatenate(raw_locations, axis=0)

    # if VERBOSE:
    #     print(len(data), len(xs), len(run_labels), raw_locations.shape)

    return np.nan_to_num(data), xs, zs, run_labels, trial_labels, raw_locations


def load_raw_joystick_labels(subject_id, run, shift_by=2):
    mask_arr = nib.load(
        f'{DATA_PATH}/{subject_id}/reference/mask.nii.gz'
    ).get_fdata()
    _, ti, xi, zi = helper.load_all_joystick_data(subject_id, run, mask_arr, shift_by)
    return np.array((xi[ti >= 0], zi[ti >= 0])).T


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_tphate_embedding(subject_id, voxel_data, ndim=20, rerun=True):
    temp_fn = os.path.join(SCRATCH_PATH, 'joystick_analyses',
                           f'{subject_id}_{ndim}d_TPHATE_embedding.npy')
    if os.path.exists(temp_fn) and not rerun:
        return np.load(temp_fn)
    dst = helper.embed_tphate(voxel_data, n_components=ndim)
    np.save(temp_fn, dst)
    return dst


def get_pca_embedding(subject_id, voxel_data, ndim=20, rerun=True):
    temp_fn = os.path.join(SCRATCH_PATH, 'joystick_analyses',
                           f'{subject_id}_{ndim}d_pcs.npy')
    if os.path.exists(temp_fn) and not rerun:
        return np.load(temp_fn)
    dst = PCA(n_components=ndim).fit_transform(voxel_data)
    np.save(temp_fn, dst)
    return dst


# ---------------------------------------------------------------------------
# RSA
# ---------------------------------------------------------------------------

def run_rsa(X, target_mat):
    Ymat   = 1 - pdist(target_mat, 'euclidean')
    Xmat   = 1 - pdist(X, 'correlation')
    result = mantel.test(Xmat, Ymat, perms=NPERM, method='spearman',
                         tail='upper', ignore_nans=True)
    return result.r, result.p, result.z


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def run_figure1_analysis(subject_id, cross_validation, regression_function='himalaya',
                         n_dim=20, hyperparam_op=False):
    """Decode location from voxel, T-PHATE, and PCA at a fixed dimensionality."""
    data, x_coords, z_coords, run_labels, trial_labels, raw_locations = \
        load_joystick_data_package(subject_id)
    tphate_emb   = get_tphate_embedding(subject_id, data, ndim=n_dim)
    pca_emb      = get_pca_embedding(subject_id, data, ndim=n_dim)
    target_coords = np.nan_to_num(np.array((x_coords, z_coords)).T)

    cv_labels = run_labels if cross_validation == 'run' else trial_labels
    CV_FUNC, _ = select_cv_and_reg_funcs(regression_function)

    dfs = []
    for name, x in zip(['voxel', 'tphate', 'pca'], [data, tphate_emb, pca_emb]):
        accs    = CV_FUNC(x, target_coords, cv_labels, hyperparam_op)
        r, p, z = run_rsa(x, raw_locations)
        v       = len(accs) + 3
        dfs.append(pd.DataFrame({
            'subject_id':     [subject_id] * v,
            'embedding_dim':  [n_dim] * v,
            'embedding_type': name,
            'cv':             cross_validation,
            'fold':           [i for i in range(len(accs))] + ['na', 'na', 'na'],
            'metric':         [regression_function] * len(accs) + ['r', 'p', 'z'],
            'score':          accs + [r, p, z],
        }))
    return pd.concat(dfs)


def run_supp_dimensionality_analysis(subject_id, cross_validation,
                                     dim2test=None, hyperparam_op=False,
                                     regression_function='himalaya'):
    """Sweep T-PHATE, PCA  across multiple dimensionalities."""
    if dim2test is None:
        dim2test = [2, 3, 5, 10, 15, 20]
    data, x_coords, z_coords, run_labels, trial_labels, raw_locations = \
        load_joystick_data_package(subject_id)
    target_coords = np.nan_to_num(np.array((x_coords, z_coords)).T)

    cv_labels  = run_labels if cross_validation == 'run' else trial_labels
    CV_FUNC, _ = select_cv_and_reg_funcs(regression_function)

    dfs = []
    for ndim in dim2test:
        tphate_emb = get_tphate_embedding(subject_id, data, ndim=ndim)
        pca_emb    = get_pca_embedding(subject_id, data, ndim=ndim)

        for name, x in zip(['tphate', 'pca'], [tphate_emb, pca_emb]):
            accs    = CV_FUNC(x, target_coords, cv_labels, hyperparam_op)
            r, p, z = run_rsa(x, raw_locations)
            v       = len(accs) + 3
            dfs.append(pd.DataFrame({
                'subject_id':     [subject_id] * v,
                'embedding_dim':  [ndim] * v,
                'embedding_type': name,
                'cv':             cross_validation,
                'fold':           [i for i in range(len(accs))] + ['na', 'na', 'na'],
                'metric':         [regression_function] * len(accs) + ['r', 'p', 'z'],
                'score':          accs + [r, p, z],
            }))
    return pd.concat(dfs)
