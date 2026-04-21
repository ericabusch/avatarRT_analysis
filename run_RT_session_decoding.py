# run_RT_session_decoding.py
# Runs location decoding analyses on real-time (RT) BCI sessions (IM, WMP, OMP),
# then computes statistics over the results.
##
# within_session_decoding  — leave-one-run-out CV within each session (IM, WMP, OMP).
#                               Stats: pairwise comparisons of MSE across session types.
#
#
# Usage:
#   python run_RT_session_decoding.py -a within

import os
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from itertools import combinations
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True
import matplotlib.pyplot as plt
import seaborn as sns

import analysis_helpers as helper
from config import (SUB_NUMBERS, INTERMEDIATE_RESULTS_PATH, FINAL_RESULTS_PATH, PLOTS_PATH,
                    VERBOSE, NPERM, SEED, context_params, colors_main, exclude_from_neural_analyses, NBOOT)
from plotting_functions import determine_symbol, make_barplot_points_precomputed
from decoding_utils import (
    load_RT_data_package,
    get_session_ids,
    select_cv_and_reg_funcs,
)

# ---------------------------------------------------------------------------
# within-session LORO cross-validation
# ---------------------------------------------------------------------------

def within_session_decoding(subject_id, regression_function='himalaya'):
    """
    Leave-one-run-out CV within each session type (IM, WMP, OMP) independently.
    Target: normalized (x, z) location coordinates.

    Returns
    -------
    pd.DataFrame  columns: subject_id, session_type, train_session, test_session,
                            cv_type, fold, test_run, mse, regression_type
    """
    if VERBOSE:
        print(f'[within_session] {subject_id}')

    im_ses, wmp_ses, omp_ses = get_session_ids(subject_id)
    CV_FUNC, _ = select_cv_and_reg_funcs(regression_function)

    rows = []
    for label, session_id in [('IM', im_ses), ('WMP', wmp_ses), ('OMP', omp_ses)]:
        data, xs, zs, run_labels, _ = load_RT_data_package(
            subject_id, session_id, data_type='projected_data'
        )
        targets      = np.array((xs, zs)).T
        mse_per_fold = CV_FUNC(data, targets, run_labels, inner_cv=True)

        for fold_idx, (run_id, mse) in enumerate(
            zip(sorted(np.unique(run_labels)), mse_per_fold)
        ):
            rows.append({
                'subject_id':      subject_id,
                'session_type':    label,
                'train_session':   label,
                'test_session':    label,
                'cv_type':         'within_session_LORO',
                'fold':            fold_idx,
                'test_run':        int(run_id),
                'mse':             mse,
                'regression_type': regression_function,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics: within-session pairwise comparisons
# ---------------------------------------------------------------------------

def compute_within_session_stats(df, n_boot=NBOOT, out_fn=None):
    """
    Pairwise comparisons of within-session MSE across all session-type pairs
    (IM vs WMP, IM vs OMP, WMP vs OMP).

    Reports mean difference (group1 - group2), 95% bootstrap CI,
    two-sided permutation p-value, and Cohen's d.

    Parameters
    ----------
    df : pd.DataFrame
        Concatenated output of run_within_all.
    n_boot : int
    out_fn : str or None

    Returns
    -------
    pd.DataFrame
    """
    # Average across folds per subject × session type
    avg = (df
           .groupby(['subject_id', 'session_type'])['mse']
           .mean()
           .reset_index())

    session_types = ['IM', 'WMP', 'OMP']
    rows = []

    for g1_label, g2_label in combinations(session_types, 2):
        g1 = avg[avg['session_type'] == g1_label].set_index('subject_id')['mse']
        g2 = avg[avg['session_type'] == g2_label].set_index('subject_id')['mse']

        subjects = g1.index.intersection(g2.index)
        v1   = g1[subjects].values
        v2   = g2[subjects].values
        diff = v1 - v2

        mean_diff, ci_lo, ci_hi, sem = helper.bootstrap_ci(diff, n_boot=n_boot, verbose=0)
        _, p_val, _ = helper.permutation_test(np.array([v1, v2]), NPERM,
                                               alternative='two-sided')
        d = helper.cohens_d_paired(diff, verbose=0)

        rows.append({
            'comparison':  f'{g1_label} vs {g2_label}',
            'group1':      g1_label,
            'group2':      g2_label,
            'n_subjects':  len(subjects),
            'mean_g1':     np.mean(v1),
            'mean_g2':     np.mean(v2),
            'mean_diff':   mean_diff,
            'ci_lower':    ci_lo,
            'ci_upper':    ci_hi,
            'sem':         sem,
            'p_value':     p_val,
            'cohens_d':    d,
        })

    stats_df = pd.DataFrame(rows)
    stats_df['significant_0.05'] = stats_df['p_value'] < 0.05

    print('\n--- Within-session MSE: pairwise comparisons ---')
    for _, row in stats_df.iterrows():
        print(f"  {row['comparison']}:  mean_diff={row['mean_diff']:.4f}  "
              f"95%CI=[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]  "
              f"p={row['p_value']:.4f}  d={row['cohens_d']:.4f}")

    if out_fn is not None:
        stats_df.to_csv(out_fn, index=False)
        print(f'Saved within-session stats: {out_fn}')

    return stats_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_within_session(df, plots_dir=PLOTS_PATH):
    """Bar + subject-points plot of within-session MSE for IM, WMP, OMP."""
    
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_context(context_params)
    np.random.seed(SEED)

    avg = (df.groupby(['subject_id', 'session_type'])['mse']
             .mean().reset_index())

    d_wide = avg.pivot(index='subject_id', columns='session_type', values='mse').dropna()
    v_im = d_wide['IM'].values
    v_wm = d_wide['WMP'].values
    v_om = d_wide['OMP'].values

    _, p_im_0, _ = helper.permutation_test(
        np.array([v_im, np.zeros(len(v_im))]), NPERM, alternative='greater')
    _, p_wm_0, _ = helper.permutation_test(
        np.array([v_wm, np.zeros(len(v_wm))]), NPERM, alternative='greater')
    _, p_om_0, _ = helper.permutation_test(
        np.array([v_om, np.zeros(len(v_om))]), NPERM, alternative='greater')
    _, p_im_wm, _ = helper.permutation_test(
        np.array([v_im, v_wm]), NPERM, alternative='two-sided')
    _, p_im_om, _ = helper.permutation_test(
        np.array([v_im, v_om]), NPERM, alternative='two-sided')
    _, p_wm_om, _ = helper.permutation_test(
        np.array([v_wm, v_om]), NPERM, alternative='two-sided')

    fig, ax = make_barplot_points_precomputed(
        avg, 'mse', 'session_type',
        pvals_vs_0=[p_im_0, p_wm_0, p_om_0],
        pvals_pairwise=[p_im_wm, p_im_om, p_wm_om],
        ylim=[0, 1.5],
        plus_bot=0.2, plus_top=0.35,
        ylabel='MSE', xlabel='Session type',
    )
    fn = os.path.join(plots_dir, 'RT_within_session_mse.pdf')
    plt.savefig(fn, transparent=True, bbox_inches='tight', format='pdf')
    plt.close()
    print(f'Saved plot: {fn}')


# ---------------------------------------------------------------------------
# Parallel runners over all subjects
# ---------------------------------------------------------------------------

def run_within_all(subjects, regression_function='himalaya', n_jobs=16,
                   out_fn=None, stats_fn=None, plots_dir=None):
    if out_fn is None:
        out_fn = os.path.join(INTERMEDIATE_RESULTS_PATH,
                              f'RT_within_session_decoding_{regression_function}.csv')

    if os.path.isfile(out_fn):
        print(f'Loading existing within-session results: {out_fn}')
        df = pd.read_csv(out_fn, index_col=0)
    else:
        joblist = [
            delayed(within_session_decoding)(s, regression_function=regression_function)
            for s in subjects
        ]
        with Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(joblist)
        df = pd.concat(results).reset_index(drop=True)
        df.to_csv(out_fn)
        print(f'Saved within-session results: {out_fn}  ({len(df)} rows)')

    if stats_fn is None:
        stats_fn = os.path.join(FINAL_RESULTS_PATH,
                                f'RT_within_session_stats_{regression_function}.csv')
    compute_within_session_stats(df, out_fn=stats_fn)
    plot_within_session(df, plots_dir=plots_dir)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run RT session location decoding (within- and cross-session).'
    )
    parser.add_argument('-a', '--analysis', type=str, default='within')
    parser.add_argument('-r', '--regression', type=str, default='himalaya',
                        choices=['himalaya', 'linreg'],
                        help='Regression model (default: himalaya)')
    parser.add_argument('-j', '--n_jobs', type=int, default=16,
                        help='Number of parallel jobs (default: 16)')
    p = parser.parse_args()

    print(f'Analysis: {p.analysis}  |  Regression: {p.regression}  |  n_jobs: {p.n_jobs}')

    SUB_IDS = [helper.format_subid(S) for S in SUB_NUMBERS if S not in exclude_from_neural_analyses]
    run_within_all(SUB_IDS, regression_function=p.regression, n_jobs=p.n_jobs)
