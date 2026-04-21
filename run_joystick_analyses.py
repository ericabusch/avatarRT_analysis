# run_joystick_analyses.py
# Runs all joystick-session location decoding analyses using himalaya ridge regression
# with leave-one-run-out cross-validation, then computes statistics.
#
# Analysis 1 (figure1):      decodes location from voxel, T-PHATE, and PCA at a
#                             fixed dimensionality (default: 20).
#                             Stats: voxel vs T-PHATE for MSE and RSA Z-score.
# Analysis 2 (dimensionality): sweeps T-PHATE and PCA across multiple dims.
#                             No statistics computed.
#
# Usage:
#   python run_joystick_analyses.py -a figure1
#   python run_joystick_analyses.py -a dimensionality
#   python run_joystick_analyses.py -a all

import os
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True
import matplotlib.pyplot as plt
import seaborn as sns

import analysis_helpers as helper
from config import (SUB_IDS, INTERMEDIATE_RESULTS_PATH, FINAL_RESULTS_PATH,
                    VERBOSE, NPERM, NBOOT, SEED, context_params)
from plotting_functions import determine_symbol
from decoding_utils import run_figure1_analysis, run_supp_dimensionality_analysis

os.makedirs(INTERMEDIATE_RESULTS_PATH, exist_ok=True)
os.makedirs(FINAL_RESULTS_PATH, exist_ok=True)
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

DIMS_TO_TEST = [2, 3, 5, 10, 15, 20]
DEFAULT_DIM  = 20
CV           = 'run'  # leave-one-run-out cross-validation



# ---------------------------------------------------------------------------
# Per-subject wrappers
# ---------------------------------------------------------------------------

def run_figure1_subject(subject_id, regression_function='himalaya', n_dim=DEFAULT_DIM):
    if VERBOSE:
        print(f'[figure1] {subject_id}')
    return run_figure1_analysis(
        subject_id,
        cross_validation=CV,
        regression_function=regression_function,
        n_dim=n_dim,
        hyperparam_op=True,
    )


def run_dimensionality_subject(subject_id, regression_function='himalaya',
                                dims_to_test=None):
    if dims_to_test is None:
        dims_to_test = DIMS_TO_TEST
    if VERBOSE:
        print(f'[dimensionality] {subject_id}')
    return run_supp_dimensionality_analysis(
        subject_id,
        cross_validation=CV,
        dim2test=dims_to_test,
        hyperparam_op=True,
        regression_function=regression_function,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_figure1(df, regression_function='himalaya', plots_dir=None):
    """Bar + subject-points plot of voxel vs T-PHATE for MSE and RSA Z-score."""
    if plots_dir is None:
        plots_dir = PLOTS_DIR
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_context(context_params)
    np.random.seed(SEED)

    raw = df[df['embedding_type'].isin(['voxel', 'tphate']) &
             df['metric'].isin([regression_function, 'z', 'r'])].copy()
    raw['score'] = pd.to_numeric(raw['score'], errors='coerce')

    tphate_subs = set(raw[raw['embedding_type'] == 'tphate']['subject_id'])
    voxel_subs  = set(raw[raw['embedding_type'] == 'voxel']['subject_id'])
    shared_subs = sorted(tphate_subs & voxel_subs)
    raw = raw[raw['subject_id'].isin(shared_subs)]

    wide = raw.pivot_table(
        index='subject_id',
        columns=['embedding_type', 'metric'],
        values='score', aggfunc='mean',
    ).reset_index()
    wide.columns = ['subject_id'] + ['_'.join(c) for c in wide.columns[1:]]

    joy_rows = []
    for _, row in wide.iterrows():
        joy_rows.append({'subject_id': row['subject_id'], 'embedding': 0,
                         'mse': row.get(f'voxel_{regression_function}', np.nan),
                         'mantel_z': row.get('voxel_z', np.nan),
                         'mantel_rho': row.get('voxel_r', np.nan)})
        joy_rows.append({'subject_id': row['subject_id'], 'embedding': 1,
                         'mse': row.get(f'tphate_{regression_function}', np.nan),
                         'mantel_z': row.get('tphate_z', np.nan),
                         'mantel_rho': row.get('tphate_r', np.nan)})
    joy_df = pd.DataFrame(joy_rows)

    pal     = [sns.color_palette('Set1', 6)[3]] * 2
    ynames  = ['mse', 'mantel_z']
    ylabels = ['Mean-squared error', 'z-score']
    yplus   = [0.5, 2]

    for yname, ylabel, yoff in zip(ynames, ylabels, yplus):
        _, ax = plt.subplots(1, 1, figsize=(3, 3))
        sns.barplot(x='embedding', y=yname, data=joy_df,
                    palette=pal, ax=ax, alpha=0.5,
                    edgecolor='k', linewidth=2, errorbar=None)
        ax.axhline(0, ls='--', c='k')

        points0, points1 = [], []
        for sub in joy_df['subject_id'].unique():
            p0 = joy_df[(joy_df['subject_id'] == sub) & (joy_df['embedding'] == 0)][yname].item()
            p1 = joy_df[(joy_df['subject_id'] == sub) & (joy_df['embedding'] == 1)][yname].item()
            points0.append(p0)
            points1.append(p1)
            ax.scatter([0, 1], [p0, p1], linewidths=1, s=60,
                       edgecolors='k', c=[pal[0], pal[1]], zorder=10)
            ax.plot([0, 1], [p0, p1], color='k', alpha=0.6, linewidth=0.4)

        arr0, arr1 = np.array(points0), np.array(points1)
        _, pv, _ = helper.permutation_test(
            np.array([arr0, arr1]), n_iterations=NBOOT, alternative='two-sided')
        pv_plot = pv if pv <= 0.5 else 1 - pv
        pstr = determine_symbol(pv_plot)
        yloc = np.max(np.concatenate((points0, points1))) + yoff
        if pstr is not None:
            ax.axhline(y=yloc, xmin=0.25, xmax=0.75, color='k', lw=1)
            ax.text(x=0.5, y=yloc + (yoff * 0.005), s=pstr, ha='center')

        ax.set(xticklabels=['voxel', 'T-PHATE'], ylabel=ylabel, xlabel='')
        sns.despine()
        plt.tight_layout()
        fn = os.path.join(plots_dir, f'joystick_{yname}.pdf')
        plt.savefig(fn, transparent=True, bbox_inches='tight', format='pdf')
        plt.close()
        print(f'Saved plot: {fn}')


# ---------------------------------------------------------------------------
# Statistics: voxel vs T-PHATE (MSE and RSA Z)
# ---------------------------------------------------------------------------

def compute_figure1_stats(df, regression_function='himalaya', n_boot=NBOOT, out_fn=None):
    """
    Compares voxel vs T-PHATE decoding performance across subjects.
    Reports mean difference (voxel - tphate), 95% bootstrap CI, permutation
    p-value (two-sided), and Cohen's d for both MSE and RSA Z-score.
    No statistics are computed for PCA vs T-PHATE.

    Parameters
    ----------
    df : pd.DataFrame
        Concatenated output of run_figure1_all.
    regression_function : str
        Label used in the 'metric' column for MSE rows.
    n_boot : int
        Bootstrap resamples for CI estimation.
    out_fn : str or None
        If provided, save the stats DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
    """
    rows = []

    for metric_name, metric_filter in [('mse', regression_function), ('rsa_z', 'z')]:
        sub_df = df[df['metric'] == metric_filter].copy()
        sub_df['score'] = pd.to_numeric(sub_df['score'], errors='coerce')

        # Average across folds per subject × embedding type
        avg = (sub_df
               .groupby(['subject_id', 'embedding_type'])['score']
               .mean()
               .reset_index())

        voxel_s  = avg[avg['embedding_type'] == 'voxel' ].set_index('subject_id')['score']
        tphate_s = avg[avg['embedding_type'] == 'tphate'].set_index('subject_id')['score']

        subjects = voxel_s.index.intersection(tphate_s.index)
        v = voxel_s[subjects].values
        t = tphate_s[subjects].values
        diff = v - t   # positive = voxel worse (higher MSE) or lower Z than T-PHATE

        mean_diff, ci_lo, ci_hi, sem = helper.bootstrap_ci(diff, n_boot=n_boot, verbose=0)
        _, p_val, _ = helper.permutation_test(np.array([v, t]), NPERM, alternative='two-sided')
        d = helper.cohens_d_paired(diff, verbose=0)

        rows.append({
            'comparison':  'voxel vs tphate',
            'metric':      metric_name,
            'n_subjects':  len(subjects),
            'mean_voxel':  np.mean(v),
            'mean_tphate': np.mean(t),
            'mean_diff':   mean_diff,
            'ci_lower':    ci_lo,
            'ci_upper':    ci_hi,
            'sem':         sem,
            'p_value':     p_val,
            'cohens_d':    d,
        })

    stats_df = pd.DataFrame(rows)
    stats_df['significant_0.05'] = stats_df['p_value'] < 0.05

    print('\n--- Figure 1: voxel vs T-PHATE ---')
    for _, row in stats_df.iterrows():
        print(f"  [{row['metric']}]  mean_diff={row['mean_diff']:.4f}  "
              f"95%CI=[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]  "
              f"p={row['p_value']:.4f}  d={row['cohens_d']:.4f}")

    if out_fn is not None:
        stats_df.to_csv(out_fn, index=False)
        print(f'Saved figure1 stats: {out_fn}')

    return stats_df


# ---------------------------------------------------------------------------
# Parallel runners over all subjects
# ---------------------------------------------------------------------------

def run_figure1_all(subjects, regression_function='himalaya', n_dim=DEFAULT_DIM,
                    n_jobs=16, out_fn=None, stats_fn=None, plots_dir=None):
    if out_fn is None:
        out_fn = os.path.join(
            INTERMEDIATE_RESULTS_PATH,
            f'joystick_figure1_{regression_function}_{CV}_cross_validation.csv',
        )

    if os.path.isfile(out_fn):
        print(f'Loading existing figure1 results: {out_fn}')
        df = pd.read_csv(out_fn, index_col=0)
    else:
        joblist = [
            delayed(run_figure1_subject)(s, regression_function=regression_function,
                                         n_dim=n_dim)
            for s in subjects
        ]
        with Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(joblist)
        df = pd.concat(results).reset_index(drop=True)
        df.to_csv(out_fn)
        print(f'Saved figure1 results: {out_fn}  ({len(df)} rows)')

    if stats_fn is None:
        stats_fn = os.path.join(
            FINAL_RESULTS_PATH,
            f'joystick_figure1_stats_{regression_function}.csv',
        )
    compute_figure1_stats(df, regression_function=regression_function, out_fn=stats_fn)
    plot_figure1(df, regression_function=regression_function, plots_dir=plots_dir)

    return df


def run_dimensionality_all(subjects, regression_function='himalaya', dims_to_test=None,
                            n_jobs=16, out_fn=None):
    if dims_to_test is None:
        dims_to_test = DIMS_TO_TEST

    if out_fn is None:
        out_fn = os.path.join(
            INTERMEDIATE_RESULTS_PATH,
            f'joystick_dimensionality_{regression_function}_{CV}_cross_validation.csv',
        )

    if os.path.isfile(out_fn):
        print(f'Loading existing dimensionality results: {out_fn}')
        return pd.read_csv(out_fn, index_col=0)

    joblist = [
        delayed(run_dimensionality_subject)(s, regression_function=regression_function,
                                            dims_to_test=dims_to_test)
        for s in subjects
    ]
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(joblist)
    df = pd.concat(results).reset_index(drop=True)
    df.to_csv(out_fn)
    print(f'Saved dimensionality results: {out_fn}  ({len(df)} rows)')
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run joystick location decoding analyses with himalaya ridge regression.'
    )
    parser.add_argument('-a', '--analysis', type=str, default='all',
                        choices=['figure1', 'dimensionality', 'all'],
                        help='Which analysis to run (default: all)')
    parser.add_argument('-r', '--regression', type=str, default='himalaya',
                        choices=['himalaya', 'linreg'],
                        help='Regression model (default: himalaya)')
    parser.add_argument('-d', '--n_dims', type=int, default=DEFAULT_DIM,
                        help='Number of dimensions for figure1 analysis (default: 20)')
    parser.add_argument('-j', '--n_jobs', type=int, default=16,
                        help='Number of parallel jobs (default: 16)')
    p = parser.parse_args()

    print(f'Analysis: {p.analysis}  |  Regression: {p.regression}  |  '
          f'CV: {CV}  |  n_jobs: {p.n_jobs}')

    if p.analysis in ('figure1', 'all'):
        run_figure1_all(SUB_IDS, regression_function=p.regression,
                        n_dim=p.n_dims, n_jobs=p.n_jobs)

    if p.analysis in ('dimensionality', 'all'):
        run_dimensionality_all(SUB_IDS, regression_function=p.regression,
                               dims_to_test=DIMS_TO_TEST, n_jobs=p.n_jobs)
