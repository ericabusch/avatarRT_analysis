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
from plotting_functions import make_barplot_points
from config import *

RESULTS_PUBLIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'results_public')
PLOTS_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'plots')
DECODING_FN    = os.path.join(RESULTS_PUBLIC, 'decoding_results_aug6_cross_session_run_cross_validation.csv')

os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_context(context_params)


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

    m_im, lo_im, hi_im   = helper.bootstrap_ci(v_im, n_boot=10000)
    m_wm, lo_wm, hi_wm   = helper.bootstrap_ci(v_wm, n_boot=10000)
    m_om, lo_om, hi_om   = helper.bootstrap_ci(v_om, n_boot=10000)
    _, p_im_0,  _ = helper.permutation_test(np.array([v_im, np.zeros(len(v_im))]),  10000, alternative='greater')
    _, p_wm_0,  _ = helper.permutation_test(np.array([v_wm, np.zeros(len(v_wm))]),  10000, alternative='greater')
    _, p_om_0,  _ = helper.permutation_test(np.array([v_om, np.zeros(len(v_om))]),  10000, alternative='greater')
    _, p_im_wm, _ = helper.permutation_test(np.array([v_im, v_wm]), 10000, alternative='two-sided')
    _, p_im_om, _ = helper.permutation_test(np.array([v_im, v_om]), 10000, alternative='two-sided')
    _, p_wm_om, _ = helper.permutation_test(np.array([v_wm, v_om]), 10000, alternative='two-sided')
    m_im_wm,  lo_im_wm,  hi_im_wm  = helper.bootstrap_ci(v_im - v_wm, n_boot=10000)
    m_im_om,  lo_im_om,  hi_im_om  = helper.bootstrap_ci(v_im - v_om, n_boot=10000)
    m_wm_om,  lo_wm_om,  hi_wm_om  = helper.bootstrap_ci(v_wm - v_om, n_boot=10000)
    d_im_wm = helper.cohens_d_paired(v_im - v_wm)
    d_im_om = helper.cohens_d_paired(v_im - v_om)
    d_wm_om = helper.cohens_d_paired(v_wm - v_om)

    print('\n--- Within-session MSE: bootstrap CIs, permutation tests, Cohen\'s d ---')
    print(f'IM:   mean={m_im:.4f}  95%CI=[{lo_im:.4f},{hi_im:.4f}]  p_vs_0={p_im_0:.4f}')
    print(f'WMP:  mean={m_wm:.4f}  95%CI=[{lo_wm:.4f},{hi_wm:.4f}]  p_vs_0={p_wm_0:.4f}')
    print(f'OMP:  mean={m_om:.4f}  95%CI=[{lo_om:.4f},{hi_om:.4f}]  p_vs_0={p_om_0:.4f}')
    print(f'IM  vs WMP:  mean={m_im_wm:.4f}  p={p_im_wm:.4f}  95%CI=[{lo_im_wm:.4f},{hi_im_wm:.4f}]  d={d_im_wm:.4f}')
    print(f'IM  vs OMP:  mean={m_im_om:.4f}  p={p_im_om:.4f}  95%CI=[{lo_im_om:.4f},{hi_im_om:.4f}]  d={d_im_om:.4f}')
    print(f'WMP vs OMP:  mean={m_wm_om:.4f}  p={p_wm_om:.4f}  95%CI=[{lo_wm_om:.4f},{hi_wm_om:.4f}]  d={d_wm_om:.4f}')

    out_fn = os.path.join(PLOTS_DIR, 'decoders_within_session.pdf')
    make_barplot_points(
        df_cong_avg, 'mse', 'session_type',
        exclude_subs=[9, 20],
        ylim=[0, 1.5],
        outfn=out_fn,
        title='Decoders evaluated within session',
        plus_bot=0.2, plus_top=0.35,
        n_iter=10000,
        sample_alternative='greater',
        pairwise_alternative='two-sided',
        ylabel='MSE', xlabel='Session type',
    )
    print(f'Saved plot: {out_fn}')


if __name__ == '__main__':
    main()
