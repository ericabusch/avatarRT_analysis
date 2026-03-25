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
#   Skipped gracefully if either file is absent.

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True
import matplotlib.pyplot as plt
import seaborn as sns
import analysis_helpers as helper
from plotting_functions import make_barplot_points, determine_symbol
from config import *

RESULTS_PUBLIC   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'results_public')
FINAL_RESULTS    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'final_results')
INTERMEDIATE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'intermediate_results')
PLOTS_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'plots')

RESAMPLING_FN    = os.path.join(RESULTS_PUBLIC, 'random_resampling_components.csv')
EVR_RUNCHANGE_FN = os.path.join(FINAL_RESULTS,  'runwise_component_EVR_neural_analysis_run_change_control.csv')
DECODING_FN      = os.path.join(RESULTS_PUBLIC, 'decoding_results_aug6_cross_session_run_cross_validation.csv')
DELTA_DECODING_FN = os.path.join(RESULTS_PUBLIC, 'delta_decoding.csv')

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
    # Plot 1 — z-scored difference
    make_barplot_points(
        results, 'zscored_difference', 'session_type',
        exclude_subs=[9, 20],
        ylim=[-2.5, 3],
        outfn=os.path.join(PLOTS_DIR, 'consolidation_zscored_evr.pdf'),
        title='',
        plus_bot=0.8, plus_top=1.3,
        n_iter=10000,
        sample_alternative='greater',
        pairwise_alternative='greater',
        ylabel='Z-Score', xlabel='Session type',
    )
    print(f'Saved plot: {os.path.join(PLOTS_DIR, "consolidation_zscored_evr.pdf")}')

    # Plot 2 — null z-score of 0 (sanity check)
    make_barplot_points(
        results, 'null_difference', 'session_type',
        exclude_subs=[9, 20],
        ylim=[-0.01, 0.01],
        outfn=os.path.join(PLOTS_DIR, 'consolidation_null_evr.pdf'),
        title='',
        n_iter=10000,
        sample_alternative='greater',
        pairwise_alternative='greater',
        ylabel='z(0) of null', xlabel='Session type',
    )
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
    _, p_omp, _ = helper.permutation_test(a, 10000)
    _, p_wmp, _ = helper.permutation_test(b, 10000)
    mean_omp, lower_omp, upper_omp = helper.bootstrap_ci(a[0], n_boot=10000)
    mean_wmp, lower_wmp, upper_wmp = helper.bootstrap_ci(b[0], n_boot=10000)
    
    print(f'OMP delta_mse: p={p_omp:.4f}, mean={mean_omp:.4f}, 95% CI=[{lower_omp:.4f}, {upper_omp:.4f}]')
    print(f'WMP delta_mse: p={p_wmp:.4f}, mean={mean_wmp:.4f}, 95% CI=[{lower_wmp:.4f}, {upper_wmp:.4f}]')

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
    
    
    # print(f'OMP delta_mse: p={p_omp:.4f}, mean={mean_omp:.4f}, 95% CI=[{lower_omp:.4f}, {upper_omp:.4f}]')
    # print(f'WMP delta_mse: p={p_wmp:.4f}, mean={mean_wmp:.4f}, 95% CI=[{lower_wmp:.4f}, {upper_wmp:.4f}]')

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
