# behavioral_results.py
# Combines the analysis from behavioral_learning.py with the plotting from behav_results.ipynb.
# Reads behavioral data from trial_regressors.csv files (local data format) and
# simulated-subject results from results/results_public/.

import numpy as np
import pandas as pd
import os, sys, glob, argparse
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from mne.stats import permutation_cluster_1samp_test
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import analysis_helpers as helper
from plotting_functions import make_barplot_points, determine_symbol
from config import *

RESULTS_PUBLIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'results_public')
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'plots')


# ─── Data loading ────────────────────────────────────────────────────────────

def load_trial_regressors(subject_id, session_id, run):
    """Load per-run trial_regressors.csv and annotate with session/run metadata."""
    fn = f'{DATA_PATH}/{subject_id}/{session_id}/behav/{subject_id}_{session_id}_run_{run:02d}_trial_regressors.csv'
    if not os.path.isfile(fn):
        return None
    df = pd.read_csv(fn, index_col=0)
    df['session_id']     = session_id
    df['session_number'] = int(session_id.split('_')[1])
    df['run_number']     = run
    df['subject_id']     = subject_id
    df.rename(columns={'trial': 'round_number', 'error_rate': 'error'}, inplace=True)
    return df


# ─── Analysis (equivalent to behavioral_learning.py) ─────────────────────────

def normalized_delta_bc_trialseries(subject_id, subject_row):
    """
    Load per-trial behavioral data and compute brain-control normalised to the
    start of each perturbation session.  Mirrors
    behavioral_learning.py::normalized_delta_bc_trialseries, reading from
    trial_regressors.csv files instead of success_rates.csv.
    """
    im_ses  = int(subject_row['im_session'].item())
    wmp_ses = int(subject_row['wmp_session'].item())
    omp_ses = int(subject_row['omp_session'].item())
    session_perturb_mapping = {'IM': im_ses, 'WMP': wmp_ses, 'OMP': omp_ses}

    all_dfs = []
    for ses_name, ses_num in session_perturb_mapping.items():
        ses_id = f'ses_{ses_num:02d}'
        for run in SESSION_TYPES_RUNS[ses_name]:
            df_run = load_trial_regressors(subject_id, ses_id, run)
            if df_run is not None:
                all_dfs.append(df_run)

    if not all_dfs:
        print(f'{subject_id}: no trial data found')
        return None

    df = pd.concat(all_dfs).reset_index(drop=True)
    df['brain_control'] = df['brain_control'].astype(float)

    # Normalise BC relative to trial-0 of the first run of each perturbation session
    normalized = np.zeros(len(df))
    for p, s in session_perturb_mapping.items():
        ses_id_str   = f'ses_{s:02d}'
        starting_run = SESSION_TYPES_RUNS[p][0]
        mask_first   = (
            (df['session_id']  == ses_id_str) &
            (df['run_number']  == starting_run) &
            (df['round_number'] == 0)
        )
        if mask_first.sum() == 0:
            continue
        normalize_to = df[mask_first]['brain_control'].values[0]
        mask_ses     = df['session_id'] == ses_id_str
        normalized[mask_ses] = df.loc[mask_ses, 'brain_control'].values - normalize_to

    df['brain_control_normalized_perturb_start'] = normalized
    outfn = f'{INTERMEDIATE_RESULTS_PATH}/{subject_id}_trialwise_behavioral_results.csv'
    df.to_csv(outfn)
    return df


def delta_BC_session(subject_id, subject_row, trialseries_df):
    """
    Compute session-level delta_BC (last trial - first trial) and delta_error.
    Mirrors behavioral_learning.py::delta_BC_session.
    """
    im_ses  = int(subject_row['im_session'].item())
    wmp_ses = int(subject_row['wmp_session'].item())
    omp_ses = int(subject_row['omp_session'].item())
    session_perturb_mapping = {'IM': im_ses, 'WMP': wmp_ses, 'OMP': omp_ses}

    rows = []
    for p, s in session_perturb_mapping.items():
        ses_id_str = f'ses_{s:02d}'
        first_run  = SESSION_TYPES_RUNS[p][0]
        final_run  = SESSION_TYPES_RUNS[p][-1]

        mask_first = (trialseries_df['session_id'] == ses_id_str) & (trialseries_df['run_number'] == first_run)
        mask_last  = (trialseries_df['session_id'] == ses_id_str) & (trialseries_df['run_number'] == final_run)
        if mask_first.sum() == 0 or mask_last.sum() == 0:
            continue

        first_df = trialseries_df[mask_first].sort_values('round_number')
        last_df  = trialseries_df[mask_last].sort_values('round_number')

        rows.append({
            'delta_error':    last_df['error'].mean()        - first_df['error'].mean(),
            'delta_BC':       last_df['brain_control'].values[-1] - first_df['brain_control'].values[0],
            'session_type':   p,
            'session_number': s,
            'subject_id':     subject_id,
        })

    return pd.DataFrame(rows)


# ─── Trial-number flattening  ──────────────────────

def flatten_trial_numbers(trialwise_df):
    """Add a flattened_trials column """
    result = trialwise_df.copy()
    result['flattened_trials'] = np.zeros(len(result)) - 1

    for sub in sorted(result.subject_id.unique()):
        a = result[result['subject_id'] == sub]
        for ses in sorted(a.session_number.unique()):
            b = a[a['session_number'] == ses]
            if ses != 2:
                b   = b[b['run_number'] > 1]
                idx = result[
                    (result['subject_id']    == sub) &
                    (result['session_number'] == ses) &
                    (result['run_number']     != 1)
                ].index
            else:
                idx = result[
                    (result['subject_id']    == sub) &
                    (result['session_number'] == ses)
                ].index
            try:
                f = helper.flatten_trials(b.round_number.values)
            except Exception:
                f = np.full(len(idx), np.nan)
            result.loc[idx, 'flattened_trials'] = f

    return result


# ─── Significance helper   ──────────────────────────

def expand_pval_for_plot(is_cluster_mask, cluster_pv, num_trials):
    is_signif_mask = np.zeros(num_trials)
    pvalues        = np.ones(num_trials)
    for slicee in is_cluster_mask:
        sl = slicee[0]
        is_signif_mask[sl]  = 1
        pvalues[sl]        *= cluster_pv[0]
    return is_signif_mask, pvalues


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Add argparse for running plotting or just analysis
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", '--verbose', type=int, default=1)
    parser.add_argument("-p", '--plot', type=int, default=1)
    p = parser.parse_args()
    PLOT = p.plot
    VERBOSE = p.verbose

    sns.set_context(context_params)

    for d in [INTERMEDIATE_RESULTS_PATH, FINAL_RESULTS_PATH, PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)

    # ── Part 1: run the behavioral analysis ──────────────────────────────────
    cumulative_info   = helper.load_info_file()
    timeseries_dfs, overall_dfs = [], []

    for subject in SUB_IDS:
        row = cumulative_info[cumulative_info['subject_ID'] == subject]
        d   = normalized_delta_bc_trialseries(subject, row)
        if d is None:
            continue
        timeseries_dfs.append(d)
        overall_dfs.append(delta_BC_session(subject, row, d))

    trialwise_results = pd.concat(timeseries_dfs).reset_index(drop=True)
    run_results       = pd.concat(overall_dfs).reset_index(drop=True)

    # Flatten trial numbers and save
    trialwise_results = flatten_trial_numbers(trialwise_results)
    trialwise_results.to_csv(BEHAV_TRIALSERIES)
    run_results.to_csv(BEHAV_SESSION_RES)
    print(f'Saved {BEHAV_TRIALSERIES}  shape={trialwise_results.shape}')
    print(f'Saved {BEHAV_SESSION_RES}  shape={run_results.shape}')

    # Save a summary statistics CSV alongside the trial series
    summary_rows = []
    for p in ORDER:
        vals = run_results[run_results['session_type'] == p]['delta_BC'].values
        summary_rows.append({
            'session_type': p,
            'n': len(vals),
            'mean_delta_BC': np.mean(vals),
            'sem_delta_BC':  vals.std(ddof=1) / np.sqrt(len(vals)),
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(FINAL_RESULTS_PATH, 'behavioral_summary_stats.csv'), index=False)

    # ── Part 2: load simulated data ──────────────────────────────
    trialwise_sim = pd.read_csv(
        f'{RESULTS_PUBLIC}/behavioral_change_trialseries_with_simulations.csv', index_col=0)
    runwise_sim   = pd.read_csv(
        f'{RESULTS_PUBLIC}/behavioral_change_runwise_with_simulations.csv',    index_col=0)

    trialwise_sim = flatten_trial_numbers(trialwise_sim)
    trialwise_sim.to_csv(BEHAV_TRIAL_W_SIM)

    # Filter subjects excluded from neural analyses (convert to string IDs)
    exclude_str = [f'avatarRT_sub_{s:02d}' for s in exclude_from_neural_analyses]
    runwise_sim_filt      = runwise_sim[~runwise_sim['subject_id'].isin(exclude_str)]
    trialwise_sim_filt    = trialwise_sim[~trialwise_sim['subject_id'].isin(exclude_str)]
    run_results_filt      = run_results[~run_results['subject_id'].isin(exclude_str)]
    trialwise_results_filt = trialwise_results[~trialwise_results['subject_id'].isin(exclude_str)]

    maxx = 56
    np.random.seed(SEED)

    # Collect all pairwise statistics as we go
    all_stats = []

    # if PLOT:
    # ── Part 3: plots ─────────────────────────────────────────────────────────

    # Plot 1 – bars: observed vs simulated delta_BC, one panel per session type
    for i, M in enumerate(ORDER):
        fig, axes = plt.subplots(1, 1, figsize=(2, 4))
        temp = runwise_sim_filt[runwise_sim_filt['session_type'] == M]
        g = sns.barplot(
            ax=axes, data=temp, x='simulated', y='delta_BC',
            palette=[colors_main[M], color_gray],
            edgecolor='k', errcolor='black', errwidth=2, alpha=0.85,
        )
        axes.axhline(0, ls='--', c='k')
        ysim = temp[temp['simulated'] == True]['delta_BC'].values
        yobs = temp[temp['simulated'] == False]['delta_BC'].values

        # Individual data points with jitter
        jitter = 0.07
        xpos_obs = np.zeros(len(yobs))  + np.random.uniform(-jitter, jitter, len(yobs))
        xpos_sim = np.ones(len(ysim))   + np.random.uniform(-jitter, jitter, len(ysim))
        axes.scatter(xpos_obs, yobs, color=colors_sim[M], edgecolors='k',
                     zorder=10, linewidths=0.6, s=20, alpha=0.8)
        axes.scatter(xpos_sim, ysim, color=color_gray, edgecolors='k',
                     zorder=10, linewidths=0.6, s=20, alpha=0.8)

        if i == 0:
            g.set(ylim=[-0.1, 0.65], xticklabels=['Obs', 'Sim'],
                  yticks=[0, 0.3, 0.6], xlabel='', ylabel='ΔBrain Control')
        else:
            g.set(ylim=[-0.1, 0.65], xticklabels=['Obs', 'Sim'],
                  yticklabels=[], xlabel='', ylabel='')
        axes.set_title(M, fontsize=10)

        _, p_obs_vs_sim = ttest_ind(yobs, ysim, permutations=10000, alternative='two-sided')
        _, p_sim_vs_0, _ = helper.permutation_test(np.array([ysim, np.zeros(len(ysim))]), 10000, alternative='two-sided')
        _, p_obs_vs_0, _ = helper.permutation_test(np.array([yobs, np.zeros(len(yobs))]), 10000, alternative='greater')
        _, lower_sim_vs_0, upper_sim_vs_0 = helper.bootstrap_ci(ysim, n_boot=10000)
        _, lower_obs_vs_0, upper_obs_vs_0 = helper.bootstrap_ci(yobs, n_boot=10000)
        symb = determine_symbol(p_obs_vs_sim)
        axes.axhline(y=0.58, xmin=0.25, xmax=0.75, ls='-', c='k')
        axes.text(x=0.5, y=0.6, s=symb)
        sns.despine(right=True, top=True)
        plt.savefig(os.path.join(PLOTS_DIR, f'true_v_sim_bars_{M}.pdf'),
                    transparent=True, bbox_inches='tight', format='pdf')
        

        all_stats.append({
            'comparison': f'{M}: obs_vs_sim_delta_BC',
            'test': 'permutation_test (n_iter=10000 )',
            'group1': 'observed', 'group2': 'simulated',
            'n1': len(yobs), 'n2': len(ysim),
            'mean1': np.mean(yobs), 'mean2': np.mean(ysim),
            'p_value': p_obs_vs_sim,
        })
        all_stats.append({
            'comparison': f'{M}: sim_delta_BC_vs_0',
            'test': 'permutation_test (n_iter=10000 )',
            'group1': 'simulated', 'group2': '0 (null)',
            'n1': len(ysim), 'n2': np.nan,
            'mean1': np.mean(ysim), 'mean2': 0,
            'p_value': p_sim_vs_0,
            'ci_lower': lower_sim_vs_0, 'ci_upper': upper_sim_vs_0,
        })
        all_stats.append({
            'comparison': f'{M}: obs_delta_BC_vs_0',
            'test': 'permutation_test (n_iter=10000 )',
            'group1': 'observed', 'group2': '0 (null)',
            'n1': len(yobs), 'n2': np.nan,
            'mean1': np.mean(yobs), 'mean2': 0,
            'p_value': p_obs_vs_0,
            'ci_lower': lower_obs_vs_0, 'ci_upper': upper_obs_vs_0,
        })

    # Plot 2 – behavioral learning curves per perturbation type (real + simulated)
    for M in ORDER:
        fig, ax = plt.subplots(1, 1, figsize=(3, 4))
        df_here = trialwise_sim_filt[
            (trialwise_sim_filt['perturbation']    == M) &
            (trialwise_sim_filt['flattened_trials'] >= 0) &
            (trialwise_sim_filt['flattened_trials'] <= maxx)
        ]
        hues = {True: color_gray, False: colors_main[M]}
        g = sns.lineplot(
            hue='simulated', x='flattened_trials',
            y='brain_control_normalized_perturb_start',
            data=df_here, palette=hues, legend=False,
        )
        g.set(xlabel='', ylabel='', xlim=[0, maxx],
              xticks=np.arange(0, 55, 25),
              ylim=[-0.1, 0.65], yticks=np.arange(0, 0.65, 0.3),
              yticklabels=[0, 30, 60])
        g.axhline(0, c='k', ls='--')
        sns.despine()
        plt.savefig(os.path.join(PLOTS_DIR, f'behavioral_curves_{M}.pdf'),
                    transparent=True, bbox_inches='tight', format='pdf')

    # ── Permutation cluster tests on real-subject trial series ────────────────
    # Use the pre-computed (HPC) observed-subject data from trialwise_sim to
    # match the notebook analysis, which found clusters in WMP and OMP.
    pvalues_trialseries = {}
    signif_mask_trialseries = {}
    trialwise_obs = trialwise_sim_filt[trialwise_sim_filt['simulated'] == 0]

    n_subs = len(trialwise_obs.subject_id.unique())
    arr_all = np.empty((n_subs, maxx + 1, len(ORDER)))

    for i, M in enumerate(ORDER):
        d = trialwise_obs[trialwise_obs['perturbation'] == M]
        counter = 0
        for s in d.subject_id.unique():
            if int(s.replace('avatarRT_sub_', '')) in exclude_from_neural_analyses:
                continue
            sd = d[
                (d['subject_id']       == s) &
                (d['flattened_trials'] >= 0) &
                (d['flattened_trials'] <= maxx)
            ]['brain_control_normalized_perturb_start'].values
            if len(sd) == 0:
                continue
            if sd.shape[0] <= maxx:
                sd = np.pad(sd, (0, maxx + 1 - sd.shape[0]), mode='edge')
            arr_all[counter, :, i] = sd[:maxx + 1]
            counter += 1

        arr = arr_all[:counter, :, i]
        t_obs, is_cluster_mask, cluster_pv, H0_perm = permutation_cluster_1samp_test(
            arr, tail=1, n_permutations=10000, threshold=2.5,
            max_step=4, adjacency=None, out_type='mask', buffer_size=None,
        )
        is_signif_mask, pvalues = expand_pval_for_plot(is_cluster_mask, cluster_pv, maxx + 1)
        pvalues_trialseries[M]     = pvalues
        signif_mask_trialseries[M] = is_signif_mask
        print(f'{M}: {int(np.sum(np.where(pvalues < 0.05, 1, 0)))} significant timepoints, {len(is_cluster_mask)} clusters')

    # Save cluster p-values to results_public
    pd.DataFrame({M: pvalues_trialseries[M] for M in ORDER}).to_csv(
        os.path.join(RESULTS_PUBLIC, 'behavioral_cluster_pvalues.csv'))

    # Plot 3 – combined learning curves with significance markers
    df_here = trialwise_results_filt[
        (trialwise_results_filt['flattened_trials'] >= 0) &
        (trialwise_results_filt['flattened_trials'] <= maxx)
    ]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    g = sns.lineplot(
        hue='perturbation', x='flattened_trials',
        y='brain_control_normalized_perturb_start',
        data=df_here, palette=colors_main, legend=False, ax=ax,
    )
    g.set(xlabel='', ylabel='', xlim=[0, maxx + 1],
          xticks=np.arange(0, maxx + 1, 25),
          ylim=[-0.3, 0.7], yticks=np.arange(-0.3, 0.65, 0.3),
          yticklabels=[-30, 0, 30, 60])
    g.axhline(0, c='k', ls='--')

    YVALS = [60, 58, 56]
    for i, M in enumerate(ORDER):
        mask  = np.where(pvalues_trialseries[M] < 0.05, 1, 0).astype(bool)
        xaxis = np.arange(0, maxx + 1)
        ax.scatter(x=xaxis[mask], y=np.repeat(YVALS[i] / 100, np.sum(mask)),
                   marker='_', s=40, c=colors_main[M])
        print(f'{M}: {np.sum(mask)} significant timepoints')

    sns.despine()
    plt.savefig(os.path.join(PLOTS_DIR, 'behavioral_curves_signif.pdf'),
                transparent=True, bbox_inches='tight', format='pdf')

    # ── Pairwise permutation stats on delta_BC (replicates make_barplot_points) ─
    include_nums = [i for i in np.arange(5, 26) if i not in exclude_from_neural_analyses]
    include_ids  = [helper.format_subid(i) for i in include_nums]
    d_stats = run_results_filt[run_results_filt['subject_id'].isin(include_ids)]
    v_im = d_stats[d_stats['session_type'] == 'IM']['delta_BC'].values
    v_wm = d_stats[d_stats['session_type'] == 'WMP']['delta_BC'].values
    v_om = d_stats[d_stats['session_type'] == 'OMP']['delta_BC'].values
    zeros = np.zeros(len(v_im))  # for paired permutation tests against 0
    
    _, p_im_0, _  = helper.permutation_test(np.array([v_im, zeros]), 10000, alternative='greater')
    _, p_wmp_0, _ = helper.permutation_test(np.array([v_wm, zeros]), 10000, alternative='greater')
    _, p_omp_0, _ = helper.permutation_test(np.array([v_om, zeros]), 10000, alternative='two-sided')
    m_im_0,  lower_im_0,  upper_im_0  = helper.bootstrap_ci(v_im, n_boot=10000)
    m_wmp_0, lower_wmp_0, upper_wmp_0 = helper.bootstrap_ci(v_wm, n_boot=10000)
    m_omp_0, lower_omp_0, upper_omp_0 = helper.bootstrap_ci(v_om, n_boot=10000)

    _, p_im_wm, _ = helper.permutation_test(np.array([v_im, v_wm]), 10000, alternative='two-sided')
    _, p_im_om, _ = helper.permutation_test(np.array([v_im, v_om]), 10000, alternative='greater')
    _, p_wm_om, _ = helper.permutation_test(np.array([v_wm, v_om]), 10000, alternative='greater')
    m_im_wmp,  lower_im_wmp,  upper_im_wmp  = helper.bootstrap_ci(v_im - v_wm, n_boot=10000)
    m_wmp_omp, lower_wmp_omp, upper_wmp_omp = helper.bootstrap_ci(v_wm - v_om, n_boot=10000)
    m_im_omp, lower_im_omp, upper_im_omp = helper.bootstrap_ci(v_im - v_om, n_boot=10000)



    d_im_0  = helper.cohens_d_paired(v_im)
    d_wmp_0 = helper.cohens_d_paired(v_wm)
    d_omp_0 = helper.cohens_d_paired(v_om)
    d_im_wm = helper.cohens_d_paired(v_im - v_wm)
    d_im_om = helper.cohens_d_paired(v_im - v_om)
    d_wm_om = helper.cohens_d_paired(v_wm - v_om)

    print('\n--- delta_BC bootstrap CIs, permutation tests, and Cohen\'s d ---')
    print(f'IM   vs 0:   mean={m_im_0:.4f}   95%CI=[{lower_im_0:.4f},{upper_im_0:.4f}]  p={p_im_0:.4f}  d={d_im_0:.4f}')
    print(f'WMP  vs 0:   mean={m_wmp_0:.4f}  95%CI=[{lower_wmp_0:.4f},{upper_wmp_0:.4f}]  p={p_wmp_0:.4f}  d={d_wmp_0:.4f}')
    print(f'OMP  vs 0:   mean={m_omp_0:.4f}  95%CI=[{lower_omp_0:.4f},{upper_omp_0:.4f}]  p={p_omp_0:.4f}  d={d_omp_0:.4f}')
    print(f'IM  vs WMP:  mean={m_im_wmp:.4f}  p={p_im_wm:.4f} 95%CI=[{lower_im_wmp:.4f},{upper_im_wmp:.4f}] d={d_im_wm:.4f}')
    print(f'IM  vs OMP:  mean={m_im_omp:.4f}  p={p_im_om:.4f} 95%CI=[{lower_im_omp:.4f},{upper_im_omp:.4f}] d={d_im_om:.4f}')
    print(f'WMP vs OMP:  mean={m_wmp_omp:.4f}  p={p_wm_om:.4f} 95%CI=[{lower_wmp_omp:.4f},{upper_wmp_omp:.4f}] d={d_wm_om:.4f}')

    for label, grp, vals, p, cohd, ci_lo, ci_hi in [
        ('IM',  'IM vs 0',  v_im, p_im_0,  d_im_0,  lower_im_0,  upper_im_0),
        ('WMP', 'WMP vs 0', v_wm, p_wmp_0, d_wmp_0, lower_wmp_0, upper_wmp_0),
        ('OMP', 'OMP vs 0', v_om, p_omp_0, d_omp_0, lower_omp_0, upper_omp_0),
    ]:
        all_stats.append({
            'comparison': f'delta_BC: {grp}',
            'test': 'permutation_test (n_iter=10000)',
            'group1': label, 'group2': '0 (null)',
            'n1': len(vals), 'n2': np.nan,
            'mean1': np.mean(vals), 'mean2': 0,
            'p_value': p, 'ci_lower': ci_lo, 'ci_upper': ci_hi,
            'cohens_d': cohd,
        })
    for label, g1, g2, vals1, vals2, p, cohd in [
        ('IM vs WMP', 'IM', 'WMP', v_im, v_wm, p_im_wm, d_im_wm),
        ('IM vs OMP', 'IM', 'OMP', v_im, v_om, p_im_om, d_im_om),
        ('WMP vs OMP','WMP','OMP', v_wm, v_om, p_wm_om, d_wm_om),
    ]:
        all_stats.append({
            'comparison': f'delta_BC: {label}',
            'test': 'permutation_test (n_iter=10000)',
            'group1': g1, 'group2': g2,
            'n1': len(vals1), 'n2': len(vals2),
            'mean1': np.mean(vals1), 'mean2': np.mean(vals2),
            'p_value': p,
            'cohens_d': cohd,
        })

    # Save all statistics to results_public
    stats_df = pd.DataFrame(all_stats)
    stats_df['significant_0.05'] = stats_df['p_value'] < 0.05
    stats_fn = os.path.join(RESULTS_PUBLIC, 'behavioral_pairwise_statistics.csv')
    stats_df.to_csv(stats_fn, index=False)
    print(f'\nSaved statistics to {stats_fn}')
    print(stats_df[['comparison', 'p_value', 'significant_0.05']].to_string(index=False))

    # Plot 4 – bar plot with individual subject points and pairwise statistics
    fig, ax = make_barplot_points(
        run_results_filt, 'delta_BC', 'session_type',
        exclude_subs=exclude_from_neural_analyses,
        ylim=[-0.3, 0.7], outfn=None, title='',
        plus_bot=0.2, plus_top=0.35, n_iter=10000,
        sample_alternative='two-sided', pairwise_alternative='two-sided',
    )
    plt.savefig(os.path.join(PLOTS_DIR, 'behavioral_difference_true.pdf'),
                transparent=True, bbox_inches='tight', format='pdf')

    # ── Linear models on delta_BC ──────────────────────────────────────────────
    model_stats = []

    # Annotate run_results_filt with counterbalancing order flag
    lm_df = run_results_filt.copy()
    lm_df['wmp_first'] = lm_df['subject_id'].isin(WM_FIRST).astype(int)

    # --- Model: delta_BC ~ session_type * wmp_first (interaction) ---
    m3 = smf.ols('delta_BC ~ C(session_type, Treatment("OMP")) * wmp_first', data=lm_df).fit()
    print('\n=== Model : delta_BC ~ session_type * wmp_first (interaction) ===')
    print(m3.summary())
    f_test3 = anova_lm(m3, typ=1)
    print(f_test3)
    for row_label, row in f_test3.iterrows():
        if pd.isna(row['F']):
            continue
        model_stats.append({
            'model': 'delta_BC ~ session_type * wmp_first',
            'term': f'F-test ({row_label})',
            'F': row['F'], 'df_num': row['df'], 'df_denom': m3.df_resid,
            'p_value': row['PR(>F)'],
            'coef': np.nan, 'std_err': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan,
        })
    for term in m3.params.index:
        ci = m3.conf_int().loc[term]
        model_stats.append({
            'model': 'delta_BC ~ session_type * wmp_first',
            'term': term, 'F': np.nan, 'df_num': np.nan, 'df_denom': m3.df_resid,
            'p_value': m3.pvalues[term],
            'coef': m3.params[term], 'std_err': m3.bse[term],
            'CI_lower': ci[0], 'CI_upper': ci[1],
        })

    # --- Model: delta_BC ~ session_type (interaction) ---
    m3 = smf.ols('delta_BC ~ C(session_type)', data=lm_df).fit()
    print('\n=== Model : delta_BC ~ session_type ===')
    print(m3.summary())
    f_test3 = anova_lm(m3, typ=1)
    print(f_test3)
    for row_label, row in f_test3.iterrows():
        if pd.isna(row['F']):
            continue
        model_stats.append({
            'model': 'delta_BC ~ session_type ',
            'term': f'F-test ({row_label})',
            'F': row['F'], 'df_num': row['df'], 'df_denom': m3.df_resid,
            'p_value': row['PR(>F)'],
            'coef': np.nan, 'std_err': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan,
        })
    for term in m3.params.index:
        ci = m3.conf_int().loc[term]
        model_stats.append({
            'model': 'delta_BC ~ session_type',
            'term': term, 'F': np.nan, 'df_num': np.nan, 'df_denom': m3.df_resid,
            'p_value': m3.pvalues[term],
            'coef': m3.params[term], 'std_err': m3.bse[term],
            'CI_lower': ci[0], 'CI_upper': ci[1],
        })

    # Save model results
    model_df = pd.DataFrame(model_stats)
    model_fn = os.path.join(RESULTS_PUBLIC, 'behavioral_lm_results.csv')
    model_df.to_csv(model_fn, index=False)
    print(f'\nSaved linear model results to {model_fn}')

    print(f'\nAll plots saved to {PLOTS_DIR}')
    print('Behavioral results complete.')
