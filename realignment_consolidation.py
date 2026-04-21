# realignment_consolidation_results.py
#
# Analysis 1 — EVR resampling:
#   For each subject/session, z-score the delta-EVR of the NFB feedback component
#   relative to a null drawn from all other components.
#   Output: results/results_public/random_resampling_components.csv
#   Plots:  results/plots/consolidation_zscored_evr.pdf
#           results/plots/consolidation_null_evr.pdf
#
# Analysis 2 — Cross-session decoding:
#   IM-trained decoders evaluated across sessions: delta_mse (run 4 - run 1).
#   Requires: results/intermediate_results/decoding_results_cross_session_run_cross_validation.csv
#             results/final_results/runwise_component_EVR_neural_analysis_run_change_control.csv

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True
import matplotlib.pyplot as plt
import seaborn as sns
import analysis_helpers as helper
from plotting_functions import make_barplot_points_precomputed, determine_symbol
from config import *

FINAL_RESULTS    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'final_results')
INTERMEDIATE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'intermediate_results')
PLOTS_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'plots')

RESAMPLING_FN    = os.path.join(INTERMEDIATE, 'random_resampling_components.csv')
EVR_RUNCHANGE_FN = os.path.join(FINAL_RESULTS,  'runwise_component_EVR_neural_analysis_run_change_control.csv')
DECODING_FN      = os.path.join(INTERMEDIATE, 'decoding_results_aug6_cross_session_run_cross_validation.csv')
DELTA_DECODING_FN = os.path.join(INTERMEDIATE, 'delta_decoding.csv')

os.makedirs(PLOTS_DIR, exist_ok=True)

SUBJECTS = [s for s in SUB_IDS if s not in ['avatarRT_sub_09', 'avatarRT_sub_20']]

sns.set_context(context_params)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — EVR resampling
# ═══════════════════════════════════════════════════════════════════════════════

def compute_resampling_results(cumulative_info):
    """
    For each subject and session, compute the z-scored change in EVR of the
    NFB feedback component relative to a null drawn from all other components.

    Z-score = (delta_EVR_target_component - mean(null)) / std(null)
    where null = 10 000 random draws from the full set of component delta-EVRs.
    """
    resampling_distributions = {'IM': [], 'WMP': [], 'OMP': []}
    true_stats               = {'IM': [], 'WMP': [], 'OMP': []}

    for subject_id in SUBJECTS:
        subject_info = cumulative_info[cumulative_info['subject_ID'] == subject_id]

        im_session  = int(subject_info['im_session'].item())
        wmp_session = int(subject_info['wmp_session'].item())
        omp_session = int(subject_info['omp_session'].item())

        im_idx  = 0
        wmp_component = int(subject_info['wmp_component'].item())
        wmp_idx = 1 if wmp_component == 1 else wmp_component - 1
        omp_idx = int(subject_info['omp_component'].item()) - 1

        session_map = {
            'IM':  {'session_number': im_session,  'component_index': im_idx},
            'WMP': {'session_number': wmp_session, 'component_index': wmp_idx},
            'OMP': {'session_number': omp_session, 'component_index': omp_idx},
        }

        for ses_name, ses_info in session_map.items():
            ses_id    = f'ses_0{ses_info["session_number"]}'
            comp_idx  = ses_info['component_index']
            first_run = SESSION_TYPES_RUNS[ses_name][0]
            last_run  = SESSION_TYPES_RUNS[ses_name][-1]

            first_X = helper.load_component_data(
                subject_id, ses_id, first_run, component_number=None,
                by_trial=False, shift_by=SHIFTBY)
            final_X = helper.load_component_data(
                subject_id, ses_id, last_run, component_number=None,
                by_trial=False, shift_by=SHIFTBY)

            evr_first = helper.run_EVR(first_X)
            evr_final = helper.run_EVR(final_X)
            diff_mat  = evr_final - evr_first

            true_stats[ses_name].append(diff_mat[comp_idx])
            resampling_distributions[ses_name].append(
                np.random.choice(diff_mat, 10000))

        print(f'  done {subject_id}')

    results = pd.DataFrame(columns=[
        'subject_id', 'session_type',
        'zscored_difference', 'true_difference', 'null_difference'])

    for ses, distribs in resampling_distributions.items():
        stats = true_stats[ses]
        for i in range(len(stats)):
            ztrue = (stats[i] - np.mean(distribs[i])) / np.std(distribs[i])
            z0    = (0        - np.mean(distribs[i])) / np.std(distribs[i])
            results.loc[len(results)] = {
                'subject_id':         SUBJECTS[i],
                'session_type':       ses,
                'zscored_difference': ztrue,
                'true_difference':    stats[i],
                'null_difference':    z0,
            }

    results.to_csv(RESAMPLING_FN)
    print(f'Saved: {RESAMPLING_FN}')
    return results


def plot_resampling_results(results):
    """
    Plot 1: z-scored EVR change of NFB component vs. session type.
    Plot 2: z-score of 0 in the null distribution (sanity check).
    """
    # Compute stats for zscored_difference
    d_wide_z = results.pivot(index='subject_id', columns='session_type', values='zscored_difference').dropna()
    v_im_z = d_wide_z['IM'].values
    v_wm_z = d_wide_z['WMP'].values
    v_om_z = d_wide_z['OMP'].values
    _, p_im_z, _  = helper.permutation_test(np.array([v_im_z, np.zeros(len(v_im_z))]), 10000, alternative='greater')
    _, p_wm_z, _  = helper.permutation_test(np.array([v_wm_z, np.zeros(len(v_wm_z))]), 10000, alternative='greater')
    _, p_om_z, _  = helper.permutation_test(np.array([v_om_z, np.zeros(len(v_om_z))]), 10000, alternative='greater')
    _, p_im_wm_z, _ = helper.permutation_test(np.array([v_im_z, v_wm_z]), 10000, alternative='greater')
    _, p_im_om_z, _ = helper.permutation_test(np.array([v_im_z, v_om_z]), 10000, alternative='greater')
    _, p_wm_om_z, _ = helper.permutation_test(np.array([v_wm_z, v_om_z]), 10000, alternative='greater')
    m_im_z, lo_im_z, hi_im_z, _ = helper.bootstrap_ci(v_im_z, n_boot=10000, verbose=0)
    m_wm_z, lo_wm_z, hi_wm_z, _ = helper.bootstrap_ci(v_wm_z, n_boot=10000, verbose=0)
    m_om_z, lo_om_z, hi_om_z, _ = helper.bootstrap_ci(v_om_z, n_boot=10000, verbose=0)
    m_im_wm_z, lo_im_wm_z, hi_im_wm_z, _ = helper.bootstrap_ci(v_im_z - v_wm_z, n_boot=10000, verbose=0)
    m_im_om_z, lo_im_om_z, hi_im_om_z, _ = helper.bootstrap_ci(v_im_z - v_om_z, n_boot=10000, verbose=0)
    m_wm_om_z, lo_wm_om_z, hi_wm_om_z, _ = helper.bootstrap_ci(v_wm_z - v_om_z, n_boot=10000, verbose=0)
    d_im_z  = helper.cohens_d_paired(v_im_z, verbose=0)
    d_wm_z  = helper.cohens_d_paired(v_wm_z, verbose=0)
    d_om_z  = helper.cohens_d_paired(v_om_z, verbose=0)
    d_im_wm_z = helper.cohens_d_paired(v_im_z - v_wm_z, verbose=0)
    d_im_om_z = helper.cohens_d_paired(v_im_z - v_om_z, verbose=0)
    d_wm_om_z = helper.cohens_d_paired(v_wm_z - v_om_z, verbose=0)

    print('\n--- zscored_difference (EVR resampling) ---')
    print(f'IM   vs 0:  mean={m_im_z:.2f}  95%CI=[{lo_im_z:.2f},{hi_im_z:.2f}]  p={p_im_z:.3f}  d={d_im_z:.2f}')
    print(f'WMP  vs 0:  mean={m_wm_z:.2f}  95%CI=[{lo_wm_z:.2f},{hi_wm_z:.2f}]  p={p_wm_z:.3f}  d={d_wm_z:.2f}')
    print(f'OMP  vs 0:  mean={m_om_z:.2f}  95%CI=[{lo_om_z:.2f},{hi_om_z:.2f}]  p={p_om_z:.3f}  d={d_om_z:.2f}')
    print(f'IM vs WMP:  mean_diff={m_im_wm_z:.2f}  95%CI=[{lo_im_wm_z:.2f},{hi_im_wm_z:.2f}]  p={p_im_wm_z:.3f}  d={d_im_wm_z:.2f}')
    print(f'IM vs OMP:  mean_diff={m_im_om_z:.2f}  95%CI=[{lo_im_om_z:.2f},{hi_im_om_z:.2f}]  p={p_im_om_z:.3f}  d={d_im_om_z:.2f}')
    print(f'WMP vs OMP: mean_diff={m_wm_om_z:.2f}  95%CI=[{lo_wm_om_z:.2f},{hi_wm_om_z:.2f}]  p={p_wm_om_z:.3f}  d={d_wm_om_z:.2f}')

    evr_stats = []
    for label, vals, p, cohd, ci_lo, ci_hi, alt in [
        ('IM',  v_im_z, p_im_z, d_im_z, lo_im_z, hi_im_z, 'greater'),
        ('WMP', v_wm_z, p_wm_z, d_wm_z, lo_wm_z, hi_wm_z, 'greater'),
        ('OMP', v_om_z, p_om_z, d_om_z, lo_om_z, hi_om_z, 'greater'),
    ]:
        evr_stats.append({'comparison': f'z-EVR: {label} vs 0',
            'test': f'permutation_test (n_iter=10000, alternative={alt})',
            'group1': label, 'group2': '0 (null)', 'n1': len(vals), 'n2': np.nan,
            'mean1': np.mean(vals), 'mean2': 0, 'p_value': p, 'ci_lower': ci_lo, 'ci_upper': ci_hi, 'cohens_d': cohd})
    for label, g1, g2, g1v, g2v, diff_vals, p, cohd, ci_lo, ci_hi, alt in [
        ('IM vs WMP', 'IM', 'WMP', v_im_z, v_wm_z, v_im_z - v_wm_z, p_im_wm_z, d_im_wm_z, lo_im_wm_z, hi_im_wm_z, 'greater'),
        ('IM vs OMP', 'IM', 'OMP', v_im_z, v_om_z, v_im_z - v_om_z, p_im_om_z, d_im_om_z, lo_im_om_z, hi_im_om_z, 'greater'),
        ('WMP vs OMP','WMP','OMP', v_wm_z, v_om_z, v_wm_z - v_om_z, p_wm_om_z, d_wm_om_z, lo_wm_om_z, hi_wm_om_z, 'greater'),
    ]:
        evr_stats.append({'comparison': f'z-EVR: {label}',
            'test': f'permutation_test (n_iter=10000, alternative={alt})',
            'group1': g1, 'group2': g2, 'n1': len(g1v), 'n2': len(g2v),
            'mean1': np.mean(g1v), 'mean2': np.mean(g2v), 'p_value': p, 'ci_lower': ci_lo, 'ci_upper': ci_hi, 'cohens_d': cohd})
    evr_stats_df = pd.DataFrame(evr_stats)
    evr_stats_df['significant_0.05'] = evr_stats_df['p_value'] < 0.05
    evr_stats_fn = os.path.join(INTERMEDIATE, 'evr_resampling_stats.csv')
    evr_stats_df.to_csv(evr_stats_fn, index=False)
    print(f'Saved statistics to {evr_stats_fn}')

    # Plot 1 — z-scored difference
    fig, ax = make_barplot_points_precomputed(
        results, 'zscored_difference', 'session_type',
        pvals_vs_0=[p_im_z, p_wm_z, p_om_z],
        pvals_pairwise=[p_im_wm_z, p_im_om_z, p_wm_om_z],
        exclude_subs=[9, 20],
        ylim=[-2.5, 3],
        plus_bot=0.8, plus_top=1.3,
        ylabel='Z-Score', xlabel='Session type',
    )
    plt.savefig(os.path.join(PLOTS_DIR, 'consolidation_zscored_evr.pdf'), transparent=True, bbox_inches='tight', format='pdf')
    plt.close()
    print(f'Saved plot: {os.path.join(PLOTS_DIR, "consolidation_zscored_evr.pdf")}')

    # Plot 2 — null z-score of 0 (sanity check)
    d_wide_null = results.pivot(index='subject_id', columns='session_type', values='null_difference').dropna()
    v_im_null = d_wide_null['IM'].values
    v_wm_null = d_wide_null['WMP'].values
    v_om_null = d_wide_null['OMP'].values
    _, p_im_null, _  = helper.permutation_test(np.array([v_im_null, np.zeros(len(v_im_null))]), 10000, alternative='greater')
    _, p_wm_null, _  = helper.permutation_test(np.array([v_wm_null, np.zeros(len(v_wm_null))]), 10000, alternative='greater')
    _, p_om_null, _  = helper.permutation_test(np.array([v_om_null, np.zeros(len(v_om_null))]), 10000, alternative='greater')
    _, p_im_wm_null, _ = helper.permutation_test(np.array([v_im_null, v_wm_null]), 10000, alternative='greater')
    _, p_im_om_null, _ = helper.permutation_test(np.array([v_im_null, v_om_null]), 10000, alternative='greater')
    _, p_wm_om_null, _ = helper.permutation_test(np.array([v_wm_null, v_om_null]), 10000, alternative='greater')
    fig, ax = make_barplot_points_precomputed(
        results, 'null_difference', 'session_type',
        pvals_vs_0=[p_im_null, p_wm_null, p_om_null],
        pvals_pairwise=[p_im_wm_null, p_im_om_null, p_wm_om_null],
        exclude_subs=[9, 20],
        ylim=[-0.01, 0.01],
        ylabel='z(0) of null', xlabel='Session type',
    )
    plt.savefig(os.path.join(PLOTS_DIR, 'consolidation_null_evr.pdf'), transparent=True, bbox_inches='tight', format='pdf')
    plt.close()
    print(f'Saved plot: {os.path.join(PLOTS_DIR, "consolidation_null_evr.pdf")}')


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Cross-session decoding (IM-trained decoders)
# ═══════════════════════════════════════════════════════════════════════════════

def run_decoding_analysis():
    """
    Load cross-session decoding results. For each subject and test session type,
    compute delta_mse = MSE(run 4) - MSE(run 1) for IM-trained decoders.
    Returns df_difference or None if the decoding file is absent.
    delta_evr column is populated if the EVR run-change file is available.
    """
    if not os.path.exists(DECODING_FN):
        print(f'  Decoding file not found: {DECODING_FN}')
        print('  Skipping cross-session decoding analysis.')
        return None

    df_dec = pd.read_csv(DECODING_FN, index_col=0)
    df_dec = df_dec[~df_dec['subject_id'].isin(['avatarRT_sub_09', 'avatarRT_sub_20'])]

    # Average across folds if present
    if 'fold' in df_dec.columns:
        df_dec = (df_dec
                  .groupby(['train_session_type', 'test_session_type',
                             'test_run', 'subject_id', 'congruent'], as_index=False)
                  ['mse'].mean())

    # IM-trained decoders; swap WMP↔OMP labels in test_session_type
    df_switch = df_dec[df_dec['train_session_type'] == 'IM'].copy()
    idx1 = df_switch[df_switch['test_session_type'] == 'WMP'].index
    idx2 = df_switch[df_switch['test_session_type'] == 'OMP'].index
    df_switch.loc[idx1, 'test_session_type'] = 'OMP'
    df_switch.loc[idx2, 'test_session_type'] = 'WMP'

    # Load EVR run-change if available (optional column)
    has_evr = os.path.exists(EVR_RUNCHANGE_FN)
    if has_evr:
        df_evr = pd.read_csv(EVR_RUNCHANGE_FN, index_col=0)
        df_evr = df_evr[~df_evr['subject_id'].isin(['avatarRT_sub_09', 'avatarRT_sub_20'])]
        dfa = df_evr[df_evr['congruent'] == True]
    else:
        print(f'  EVR run-change file not found — delta_evr column will be NaN.')

    rows = []
    for sub in df_switch.subject_id.unique():
        for test_type in ['IM', 'WMP', 'OMP']:
            temp = df_switch[
                (df_switch['test_session_type'] == test_type) &
                (df_switch['subject_id'] == sub)]
            if temp.empty:
                continue
            first_run = temp[temp['test_run'] == 1]['mse'].values
            final_run = temp[temp['test_run'] == 4]['mse'].values
            if len(first_run) == 0 or len(final_run) == 0:
                continue
            evr_val = np.nan
            if has_evr:
                evr_row = dfa[(dfa['session_type'] == test_type) & (dfa['subject_id'] == sub)]
                if not evr_row.empty:
                    evr_val = evr_row['delta_evr_since_perturb'].iloc[0]
            rows.append({
                'train_session_type': 'IM',
                'test_session_type':  test_type,
                'subject_id':         sub,
                'delta_mse':          float(final_run[0]) - float(first_run[0]),
                'delta_evr':          evr_val,
            })

    return pd.DataFrame(rows)


def plot_decoding_results(df_difference):
    """
    Barplot of delta_mse for IM-trained decoders across WMP and OMP sessions,
    with one-sample permutation tests vs. 0.
    """
    a = np.array([df_difference[df_difference['test_session_type'] == 'OMP']['delta_mse'].values,
                  np.zeros(len(df_difference['subject_id'].unique()))])
    b = np.array([df_difference[df_difference['test_session_type'] == 'WMP']['delta_mse'].values,
                  np.zeros(len(df_difference['subject_id'].unique()))])
    _, p_omp, _ = helper.permutation_test(a, 10000, alternative='greater')
    _, p_wmp, _ = helper.permutation_test(b, 10000, alternative='greater')
    mean_omp, lower_omp, upper_omp, _ = helper.bootstrap_ci(a[0], n_boot=10000, verbose=0)
    mean_wmp, lower_wmp, upper_wmp, _ = helper.bootstrap_ci(b[0], n_boot=10000, verbose=0)
    d_omp = helper.cohens_d_paired(a[0], verbose=0)
    d_wmp = helper.cohens_d_paired(b[0], verbose=0)

    print(f'OMP delta_mse: p={p_omp:.2f}, mean={mean_omp:.2f}, 95% CI=[{lower_omp:.2f}, {upper_omp:.2f}], d={d_omp:.2f}')
    print(f'WMP delta_mse: p={p_wmp:.2f}, mean={mean_wmp:.2f}, 95% CI=[{lower_wmp:.2f}, {upper_wmp:.2f}], d={d_wmp:.2f}')

    dec_stats = [
        {'comparison': 'delta_MSE: OMP vs 0', 'test': 'permutation_test (n_iter=10000, alternative=greater)',
         'group1': 'OMP', 'group2': '0 (null)', 'n1': len(a[0]), 'n2': np.nan,
         'mean1': mean_omp, 'mean2': 0, 'p_value': p_omp, 'ci_lower': lower_omp, 'ci_upper': upper_omp, 'cohens_d': d_omp},
        {'comparison': 'delta_MSE: WMP vs 0', 'test': 'permutation_test (n_iter=10000, alternative=greater)',
         'group1': 'WMP', 'group2': '0 (null)', 'n1': len(b[0]), 'n2': np.nan,
         'mean1': mean_wmp, 'mean2': 0, 'p_value': p_wmp, 'ci_lower': lower_wmp, 'ci_upper': upper_wmp, 'cohens_d': d_wmp},
    ]
    dec_stats_df = pd.DataFrame(dec_stats)
    dec_stats_df['significant_0.05'] = dec_stats_df['p_value'] < 0.05
    dec_stats_fn = os.path.join(FINAL_RESULTS, 'cross_session_decoding_stats.csv')
    dec_stats_df.to_csv(dec_stats_fn, index=False)
    print(f'Saved statistics to {dec_stats_fn}')

    fig, ax = plt.subplots(1, 1, figsize=(3, 4))
    sns.barplot(data=df_difference, x='test_session_type', order=['WMP', 'OMP'],
                y='delta_mse', ax=ax,
                palette=[colors_main['WMP'], colors_main['OMP']],
                hue='test_session_type',
                hue_order=['WMP', 'OMP'],
                edgecolor='k', linewidth=2, alpha=0.7)
    sns.stripplot(data=df_difference, x='test_session_type', order=['WMP', 'OMP'],
                  y='delta_mse', ax=ax,
                  hue='test_session_type',
                  hue_order=['WMP', 'OMP'],
                  palette=[colors_main['WMP'], colors_main['OMP']],
                  edgecolor='k', linewidth=1, jitter=False)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.text(x=0 - 0.05, y=0.5, s=determine_symbol(p_wmp), size=16)
    ax.text(x=1 - 0.05, y=0.5, s=determine_symbol(p_omp), size=12)
    ax.set(title=r'$\Delta$ performance of IM-trained decoders',
           ylabel=r'$\Delta$ MSE', xlabel='Session type',
           ylim=[-0.2, 0.55])
    ax.axhline(0, color='k', ls='--')
    sns.despine()
    fig.tight_layout()

    out_fn = os.path.join(PLOTS_DIR, 'consolidation_delta_decoding.pdf')
    plt.savefig(out_fn, format='pdf', transparent=True)
    print(f'Saved plot: {out_fn}')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Analysis 1: EVR resampling ───────────────────────────────────────────
    if os.path.exists(RESAMPLING_FN):
        print('Resampling results found — loading.')
        results = pd.read_csv(RESAMPLING_FN, index_col=0)
    else:
        print('Resampling results not found — computing (requires raw data).')
        cumulative_info = helper.load_info_file()
        results = compute_resampling_results(cumulative_info)

    print(f'\nResampling results: {results.shape[0]} rows')
    plot_resampling_results(results)
    # a = results[results['session_type'] == 'IM']['true_difference'].values

    # b = results[results['session_type'] == 'WMP']['true_difference'].values
    # c = results[results['session_type'] == 'OMP']['true_difference'].values

    # _, p_omp, _ = helper.permutation_test(c, 10000)
    # _, p_wmp, _ = helper.permutation_test(b, 10000)
    # _, p_im, _ = helper.permutation_test(a, 10000)
    
    # mean_omp, lower_omp, upper_omp = helper.bootstrap_ci(c[0], n_boot=10000)
    # mean_wmp, lower_wmp, upper_wmp = helper.bootstrap_ci(b[0], n_boot=10000)
    # mean_im, lower_wmp, upper_wmp = helper.bootstrap_ci(b[0], n_boot=10000)
    
    
    # print(f'OMP delta_mse: p={p_omp:.2f}, mean={mean_omp:.2f}, 95% CI=[{lower_omp:.2f}, {upper_omp:.2f}]')
    # print(f'WMP delta_mse: p={p_wmp:.2f}, mean={mean_wmp:.2f}, 95% CI=[{lower_wmp:.2f}, {upper_wmp:.2f}]')

    # ── Analysis 2: Cross-session decoding ──────────────────────────────────
    print('\n--- Cross-session decoding ---')
    if os.path.exists(DELTA_DECODING_FN):
        print('delta_decoding.csv found — loading.')
        df_difference = pd.read_csv(DELTA_DECODING_FN, index_col=0)
    else:
        df_difference = run_decoding_analysis()
        if df_difference is not None:
            df_difference.to_csv(DELTA_DECODING_FN)
            print(f'Saved: {DELTA_DECODING_FN}')
    if df_difference is not None:
        plot_decoding_results(df_difference)


if __name__ == '__main__':
    main()
