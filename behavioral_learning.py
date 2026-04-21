# behavioral_results.py
# Combines the analysis from behavioral_learning.py with the plotting 
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
from plotting_functions import make_barplot_points_precomputed, determine_symbol
from config import *
SEED=4
RESULTS_PUBLIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'results_public')
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'plots')
np.random.seed(SEED)

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


# ─── Analysis  ─────────────────────────

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

    # Normalize BC relative to trial-0 of the first run of each perturbation session
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


def load_behavioral_change_simulations():
    if os.path.exists(f'{FINAL_RESULTS_PATH}/{BEHAV_TRIAL_W_SIM}') and os.path.exists(f'{FINAL_RESULTS_PATH}/{BEHAV_SESSION_W_SIM}'):
        FN_TRIAL = f'{FINAL_RESULTS_PATH}/{BEHAV_TRIAL_W_SIM}'
        FN_SESSION = f'{FINAL_RESULTS_PATH}/{BEHAV_SESSION_W_SIM}'

    elif os.path.exists(f'{INTERMEDIATE_RESULTS_PATH}/{BEHAV_TRIAL_W_SIM}') and os.path.exists(f'{INTERMEDIATE_RESULTS_PATH}/{BEHAV_SESSION_W_SIM}'):
        FN_TRIAL = f'{INTERMEDIATE_RESULTS_PATH}/{BEHAV_TRIAL_W_SIM}'
        FN_SESSION = f'{INTERMEDIATE_RESULTS_PATH}/{BEHAV_SESSION_W_SIM}'
    else:
        return None, None
    print(f'Found existing simulation results files {BEHAV_TRIAL_W_SIM} and {BEHAV_SESSION_W_SIM}, loading them...')
    trialwise_sim = pd.read_csv(FN_TRIAL, index_col=0)
    runwise_sim   = pd.read_csv(FN_SESSION, index_col=0)
    return trialwise_sim, runwise_sim


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

    trialwise_results, run_results = load_behavioral_change_simulations()

    # Check if trialwise and runwise are none
    if trialwise_results is not None and run_results is not None:
        print('Successfully loaded outputs .')
    else:
        print('No existing output files found, running analysis on trial_regressors.csv files...')
        timeseries_dfs, overall_dfs = [], []

        for subject in SUB_IDS + SIMULATED_SUBS:
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
        trialwise_results.to_csv({FINAL_RESULTS_PATH}/{BEHAV_TRIAL_W_SIM})
        run_results.to_csv({FINAL_RESULTS_PATH}/{BEHAV_SESSION_W_SIM})
        print(f'Saved {BEHAV_TRIALSERIES}  shape={trialwise_results.shape}')
        print(f'Saved {BEHAV_SESSION_RES}  shape={run_results.shape}')

    # Filter subjects excluded from neural analyses (convert to string IDs)
    exclude_str = [f'avatarRT_sub_{s:02d}' for s in exclude_from_neural_analyses]
    run_results_filt      = run_results[~run_results['subject_id'].isin(exclude_str)]
    trialwise_results_filt = trialwise_results[~trialwise_results['subject_id'].isin(exclude_str)]

    maxx = 56 # for plotting trial series, based on max flattened trial number across all subjects/sessions after filtering
   
    # ── Part 3: statistics ────────────────────────────────────────────────────

    all_stats = []

    # Prepare per-session real-subject delta_BC vectors
    include_nums = [i for i in np.arange(5, 26) if i not in exclude_from_neural_analyses]
    include_ids  = [helper.format_subid(i) for i in include_nums]
    # This will search for only real participants, excluding simulated ones and those excluded from neural analyses
    d_stats = run_results_filt[run_results_filt['subject_id'].isin(include_ids)] 
    v_im = d_stats[d_stats['session_type'] == 'IM']['delta_BC'].values
    v_wm = d_stats[d_stats['session_type'] == 'WMP']['delta_BC'].values
    v_om = d_stats[d_stats['session_type'] == 'OMP']['delta_BC'].values

    # --- 3a: real subjects delta_BC vs 0 (IM/WMP: greater; OMP: two-sided) ---
    _, p_im_0, _  = helper.permutation_test(np.array([v_im, np.zeros(len(v_im))]), 10000, alternative='greater')
    _, p_wmp_0, _ = helper.permutation_test(np.array([v_wm, np.zeros(len(v_wm))]), 10000, alternative='greater')
    _, p_omp_0, _ = helper.permutation_test(np.array([v_om, np.zeros(len(v_om))]), 10000, alternative='two-sided')
    m_im_0,  lower_im_0,  upper_im_0,  _ = helper.bootstrap_ci(v_im, n_boot=10000, verbose=0)
    m_wmp_0, lower_wmp_0, upper_wmp_0, _ = helper.bootstrap_ci(v_wm, n_boot=10000, verbose=0)
    m_omp_0, lower_omp_0, upper_omp_0, _ = helper.bootstrap_ci(v_om, n_boot=10000, verbose=0)
    d_im_0  = helper.cohens_d_paired(v_im, verbose=0)
    d_wmp_0 = helper.cohens_d_paired(v_wm, verbose=0)
    d_omp_0 = helper.cohens_d_paired(v_om, verbose=0)

    print('\n--- delta_BC vs 0 (real subjects) ---')
    print(f'IM   vs 0:  mean={m_im_0:.4f}  95%CI=[{lower_im_0:.4f},{upper_im_0:.4f}]  p={p_im_0:.4f}  d={d_im_0:.4f}')
    print(f'WMP  vs 0:  mean={m_wmp_0:.4f}  95%CI=[{lower_wmp_0:.4f},{upper_wmp_0:.4f}]  p={p_wmp_0:.4f}  d={d_wmp_0:.4f}')
    print(f'OMP  vs 0:  mean={m_omp_0:.4f}  95%CI=[{lower_omp_0:.4f},{upper_omp_0:.4f}]  p={p_omp_0:.4f}  d={d_omp_0:.4f}')

    for label, vals, p, cohd, ci_lo, ci_hi, alt in [
        ('IM',  v_im, p_im_0,  d_im_0,  lower_im_0,  upper_im_0,  'greater'),
        ('WMP', v_wm, p_wmp_0, d_wmp_0, lower_wmp_0, upper_wmp_0, 'greater'),
        ('OMP', v_om, p_omp_0, d_omp_0, lower_omp_0, upper_omp_0, 'two-sided'),
    ]:
        all_stats.append({
            'comparison': f'delta_BC: {label} vs 0',
            'test': f'permutation_test (n_iter=10000, alternative={alt})',
            'group1': label, 'group2': '0 (null)',
            'n1': len(vals), 'n2': np.nan,
            'mean1': np.mean(vals), 'mean2': 0,
            'p_value': p, 'ci_lower': ci_lo, 'ci_upper': ci_hi,
            'cohens_d': cohd,
        })

    # --- 3b: pairwise comparisons (IM vs WMP: two-sided; IM/WMP vs OMP: greater) ---
    _, p_im_wm, _ = helper.permutation_test(np.array([v_im, v_wm]), 10000, alternative='two-sided')
    _, p_im_om, _ = helper.permutation_test(np.array([v_im, v_om]), 10000, alternative='greater')
    _, p_wm_om, _ = helper.permutation_test(np.array([v_wm, v_om]), 10000, alternative='greater')
    m_im_wmp,  lower_im_wmp,  upper_im_wmp,  _ = helper.bootstrap_ci(v_im - v_wm, n_boot=10000, verbose=0)
    m_im_omp,  lower_im_omp,  upper_im_omp,  _ = helper.bootstrap_ci(v_im - v_om, n_boot=10000, verbose=0)
    m_wmp_omp, lower_wmp_omp, upper_wmp_omp, _ = helper.bootstrap_ci(v_wm - v_om, n_boot=10000, verbose=0)
    d_im_wm = helper.cohens_d_paired(v_im - v_wm, verbose=0)
    d_im_om = helper.cohens_d_paired(v_im - v_om, verbose=0)
    d_wm_om = helper.cohens_d_paired(v_wm - v_om, verbose=0)

    print('\n--- pairwise delta_BC comparisons ---')
    print(f'IM  vs WMP:  mean_diff={m_im_wmp:.4f}  p={p_im_wm:.4f}  95%CI=[{lower_im_wmp:.4f},{upper_im_wmp:.4f}]  d={d_im_wm:.4f}')
    print(f'IM  vs OMP:  mean_diff={m_im_omp:.4f}  p={p_im_om:.4f}  95%CI=[{lower_im_omp:.4f},{upper_im_omp:.4f}]  d={d_im_om:.4f}')
    print(f'WMP vs OMP:  mean_diff={m_wmp_omp:.4f}  p={p_wm_om:.4f}  95%CI=[{lower_wmp_omp:.4f},{upper_wmp_omp:.4f}]  d={d_wm_om:.4f}')

    for label, g1, g2, g1_vals, g2_vals, diff_vals, p, cohd, ci_lo, ci_hi, alt in [
        ('IM vs WMP', 'IM',  'WMP', v_im, v_wm, v_im - v_wm, p_im_wm, d_im_wm, lower_im_wmp,  upper_im_wmp,  'two-sided'),
        ('IM vs OMP', 'IM',  'OMP', v_im, v_om, v_im - v_om, p_im_om, d_im_om, lower_im_omp,  upper_im_omp,  'greater'),
        ('WMP vs OMP','WMP', 'OMP', v_wm, v_om, v_wm - v_om, p_wm_om, d_wm_om, lower_wmp_omp, upper_wmp_omp, 'greater'),
    ]:
        all_stats.append({
            'comparison': f'delta_BC: {label}',
            'test': f'permutation_test (n_iter=10000, alternative={alt})',
            'group1': g1, 'group2': g2,
            'n1': len(g1_vals), 'n2': len(g2_vals),
            'mean1': np.mean(g1_vals), 'mean2': np.mean(g2_vals),
            'p_value': p, 'ci_lower': ci_lo, 'ci_upper': ci_hi,
            'cohens_d': cohd,
        })

    # --- 3c: obs vs sim delta_BC (alternative=two-sided, one per session type) ---
    obs_data, sim_data, obs_vs_sim_pvals = {}, {}, {}
    print('\n--- obs vs sim delta_BC ---')
    for M in ORDER:
        temp = run_results_filt[run_results_filt['session_type'] == M]
        yobs = temp[temp['simulated'] == False]['delta_BC'].values
        ysim = temp[temp['simulated'] == True]['delta_BC'].values
        obs_data[M] = yobs
        sim_data[M] = ysim
        _, p_obs_sim = ttest_ind(yobs, ysim, permutations=10000, alternative='two-sided')
        obs_vs_sim_pvals[M] = p_obs_sim
        m_obs, lower_obs, upper_obs, _ = helper.bootstrap_ci(yobs, n_boot=10000, verbose=0)
        m_sim, lower_sim, upper_sim, _ = helper.bootstrap_ci(ysim, n_boot=10000, verbose=0)
        d_obs_sim = helper.cohens_d_independent(yobs, ysim, verbose=0)
        print(f'{M}: obs mean={m_obs:.4f} 95%CI=[{lower_obs:.4f},{upper_obs:.4f}]  '
              f'sim mean={m_sim:.4f} 95%CI=[{lower_sim:.4f},{upper_sim:.4f}]  '
              f'p={p_obs_sim:.4f}  d={d_obs_sim:.4f}')
        all_stats.append({
            'comparison': f'{M}: obs vs sim delta_BC',
            'test': 'permutation_ttest_ind (n_iter=10000, alternative=two-sided)',
            'group1': 'observed', 'group2': 'simulated',
            'n1': len(yobs), 'n2': len(ysim),
            'mean1': m_obs, 'mean2': m_sim,
            'p_value': p_obs_sim,
            'ci_lower': lower_obs, 'ci_upper': upper_obs,
            'cohens_d': d_obs_sim,
        })

    # Save statistics
    stats_df = pd.DataFrame(all_stats)
    stats_df['significant_0.05'] = stats_df['p_value'] < 0.05
    stats_fn = os.path.join(FINAL_RESULTS_PATH, 'behavioral_pairwise_statistics.csv')
    stats_df.to_csv(stats_fn, index=False)
    print(f'\nSaved statistics to {stats_fn}')
    print(stats_df[['comparison', 'p_value', 'significant_0.05']].to_string(index=False))

    # ── Part 4: plots ─────────────────────────────────────────────────────────

    # Plot 1 – bars: observed vs simulated delta_BC, one panel per session type
    for i, M in enumerate(ORDER):
        fig, axes = plt.subplots(1, 1, figsize=(2, 4))
        temp = run_results_filt[run_results_filt['session_type'] == M]
        g = sns.barplot(
            ax=axes, data=temp, x='simulated', y='delta_BC',
            palette=[colors_main[M], color_gray],
            edgecolor='k', errcolor='black', errwidth=2, alpha=0.85,
        )
        axes.axhline(0, ls='--', c='k')
        yobs = obs_data[M]
        ysim = sim_data[M]

        # Individual data points with jitter
        jitter = 0.07
        xpos_obs = np.zeros(len(yobs)) + np.random.uniform(-jitter, jitter, len(yobs))
        xpos_sim = np.ones(len(ysim))  + np.random.uniform(-jitter, jitter, len(ysim))
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

        symb = determine_symbol(obs_vs_sim_pvals[M])
        axes.axhline(y=0.58, xmin=0.25, xmax=0.75, ls='-', c='k')
        axes.text(x=0.5, y=0.6, s=symb)
        sns.despine(right=True, top=True)
        plt.savefig(os.path.join(PLOTS_DIR, f'true_v_sim_bars_{M}.pdf'),
                    transparent=True, bbox_inches='tight', format='pdf')

    # Plot 2 – behavioral learning curves per perturbation type (real + simulated)
    for M in ORDER:
        fig, ax = plt.subplots(1, 1, figsize=(3, 4))
        df_here = trialwise_results_filt[
            (trialwise_results_filt['perturbation']    == M) &
            (trialwise_results_filt['flattened_trials'] >= 0) &
            (trialwise_results_filt['flattened_trials'] <= maxx)
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

    pvalues_trialseries = {}
    signif_mask_trialseries = {}
    trialwise_obs = trialwise_results_filt[trialwise_results_filt['simulated'] == 0]

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

    # Plot 4 – bar plot with individual subject points and precomputed pairwise statistics
    fig, ax = make_barplot_points_precomputed(
        run_results_filt, 'delta_BC', 'session_type',
        pvals_vs_0=[p_im_0, p_wmp_0, p_omp_0],
        pvals_pairwise=[p_im_wm, p_im_om, p_wm_om],
        exclude_subs=exclude_from_neural_analyses,
        ylim=[-0.3, 0.7],
        plus_bot=0.2, plus_top=0.35,
    )
    plt.savefig(os.path.join(PLOTS_DIR, 'behavioral_difference_true.pdf'),
                transparent=True, bbox_inches='tight', format='pdf')

    # ── Linear models on delta_BC ──────────────────────────────────────────────
    model_stats = []

    # Annotate run_results_filt with counterbalancing order flag
    lm_df = run_results_filt.copy()
    lm_df['wmp_first'] = lm_df['subject_id'].isin(WM_FIRST).astype(int)

    def add_partial_eta_squared(anova_table):
        ss_res = anova_table.loc["Residual", "sum_sq"]
        anova_table["eta_p2"] = anova_table["sum_sq"] / (anova_table["sum_sq"] + ss_res)
        return anova_table

    # --- Model: delta_BC ~ session_type ---
    m3 = smf.ols('delta_BC ~ C(session_type)', data=lm_df).fit()
    print('\n=== Model : delta_BC ~ session_type ===')
    f_test3 = anova_lm(m3, typ=1)
    f_test3 = add_partial_eta_squared(f_test3)
    for row_label, row in f_test3.iterrows():
        if pd.isna(row['F']):
            continue
        model_stats.append({
            'model': 'delta_BC ~ session_type ',
            'term': f'F-test ({row_label})',
            'F': row['F'], 'df_num': row['df'], 'df_denom': m3.df_resid,
            'p_value': row['PR(>F)'],
            'coef': np.nan, 'std_err': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan,
            'R2': m3.rsquared, 'adj_R2': m3.rsquared_adj, 'partial_eta_squared': row['sum_sq'] / (row['sum_sq'] + m3.ssr),
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
    
    

    # --- Model: delta_BC ~ session_type * wmp_first (interaction) ---
    # Remove IM session since it always comes first
    lm_df = lm_df[lm_df['session_type'] != 'IM']
    formula = 'delta_BC ~ session_type : wmp_first + wmp_first'
    m3 = smf.ols(formula, data=lm_df).fit()
    print('\n=== Model : {formula} ===')
    f_test3 = anova_lm(m3, typ=1)
    for row_label, row in f_test3.iterrows():
        if pd.isna(row['F']):
            continue
        model_stats.append({
            'model': formula,
            'term': f'F-test ({row_label})',
            'F': row['F'], 'df_num': row['df'], 'df_denom': m3.df_resid,
            'p_value': row['PR(>F)'],
            'coef': np.nan, 'std_err': np.nan, 'CI_lower': np.nan, 'CI_upper': np.nan,
            'R2': m3.rsquared, 'adj_R2': m3.rsquared_adj, 'eta_squared': row['sum_sq'] / (row['sum_sq'] + m3.ssr),
        })
    for term in m3.params.index:
        ci = m3.conf_int().loc[term]
        model_stats.append({
            'model': formula,
            'term': term, 'F': np.nan, 'df_num': np.nan, 'df_denom': m3.df_resid,
            'p_value': m3.pvalues[term],
            'coef': m3.params[term], 'std_err': m3.bse[term],
            'CI_lower': ci[0], 'CI_upper': ci[1],
        })
    


    # Save model results
    model_df = pd.DataFrame(model_stats)
    model_fn = os.path.join(FINAL_RESULTS_PATH, 'behavioral_lm_results.csv')
    model_df.to_csv(model_fn, index=False)
    print(f'\nSaved linear model results to {model_fn}')

    print(f'\nAll plots saved to {PLOTS_DIR}')
    print('Behavioral results complete.')
