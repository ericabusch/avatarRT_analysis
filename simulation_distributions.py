# simulation_distributions.py
# For each subject × session type (IM, WMP), simulates two null models of
# behavioral change (subselection and realignment) from the decoded-angle
# distributions, then scores the true change against each null.
#
# Load-or-compute pattern:
#   - If SUMMARY_FN exists, loads it; otherwise runs all simulations and saves.
#
# Outputs (in results/final_results/):
#   - summary_results_simulations_all_final1.csv
# Outputs (in results/intermediate_results/):
#   - simulations_all_final1.csv
# Outputs (in results/plots/):
#   - simulations_normalized_density.pdf

import numpy as np
import pandas as pd
import os
import time
import itertools
import argparse
import scipy
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['pdf.use14corefonts'] = True
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_1samp

import analysis_helpers as helper
from plotting_functions import determine_symbol
from config import *

SUMMARY_FN = os.path.join(FINAL_RESULTS_PATH, 'summary_results_simulations_all_final1.csv')
SIM_FN     = os.path.join(INTERMEDIATE_RESULTS_PATH, 'simulations_all_final1.csv')

SUBJECTS   = [s for s in SUB_IDS if s not in [f'avatarRT_sub_{n:02d}' for n in exclude_from_neural_analyses]]
PROPORTIONS = [0.2, 0.3, 0.4, 0.5]

sns.set_context(context_params)


# ─── Simulation helpers ───────────────────────────────────────────────────────

def simulate_subselection(distrib, j=None, proportion=0.5):
    if j is None:
        j = np.mean(distrib)
    distances = np.abs(distrib - j)
    probabilities = 1 / (distances + 1e-6)
    probabilities /= np.sum(probabilities)
    num_points_to_select = int(len(distrib) * proportion)
    selected_indices = np.random.choice(len(distrib), size=num_points_to_select, replace=False, p=probabilities)
    distrib = distrib[np.setdiff1d(np.arange(len(distrib)), selected_indices)]
    selected_indices = np.random.choice(len(distrib), size=len(distrib) + num_points_to_select, replace=True)
    return distrib[selected_indices]


def scale_data(X, maxx=1, minn=-1, meann=None):
    if meann is not None:
        X = X - np.mean(X) + meann
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X_std * (maxx - minn) + minn


def best_fit_distribution_v2(data, bins=None, distributions=None, verbose=False):
    if distributions is None:
        distributions = [stats.norm, stats.lognorm, stats.gamma, stats.beta, stats.triang]
    data = np.asarray(data)
    distests, params = [], []
    for distribution in distributions:
        args = distribution.fit(data)
        distest = stats.kstest(data, distribution.cdf, args=args)
        distests.append(distest.statistic)
        params.append(args)
    idx = np.argmin(distests)
    return distributions[idx], params[idx]


def simulate_realigned_distribution(initial_distribution, scale_factor=1.5, size=None,
                                    bins=50, distributions=None, return_func=False):
    dist, params = best_fit_distribution_v2(initial_distribution, bins=bins,
                                            distributions=distributions, verbose=False)
    *shapes, loc, scale = params
    new_scale = scale * scale_factor
    if size is None:
        size = len(initial_distribution)
    new_data = dist.rvs(*shapes, loc=loc, scale=new_scale, size=size)
    new_data = scale_data(new_data,
                          maxx=np.max(initial_distribution),
                          minn=np.min(initial_distribution),
                          meann=np.mean(initial_distribution))
    if return_func:
        return new_data, dist
    return new_data


def run_simulations(initial_data, proportion=0.5, n_sims=1000):
    sim_subselection, sim_realignment, distribution_type = [], [], []
    for _ in range(n_sims):
        sim_subselection.append(simulate_subselection(initial_data, proportion=proportion))
        if len(distribution_type) == 0:
            r, dist = simulate_realigned_distribution(initial_data, scale_factor=1 + proportion,
                                                      size=None, bins=50, return_func=True)
            distribution_type.append(dist)
            sim_realignment.append(r)
        else:
            sim_realignment.append(simulate_realigned_distribution(
                initial_data, scale_factor=1 + proportion,
                size=None, distributions=distribution_type, bins=50))
    return np.array(sim_subselection), np.array(sim_realignment)


def compute_change_scores(pre_learning, post_learning):
    delta_var = np.var(post_learning) - np.var(pre_learning)
    pre_ent  = scipy.stats.entropy(np.histogram(pre_learning,  bins=50, density=True)[0] + 1e-6)
    post_ent = scipy.stats.entropy(np.histogram(post_learning, bins=50, density=True)[0] + 1e-6)
    return delta_var, post_ent - pre_ent


def score_simulations(subselection_sims, realignment_sims, pre_learning, post_learning):
    subsel_scores  = np.array([compute_change_scores(pre_learning, sim) for sim in subselection_sims])
    realign_scores = np.array([compute_change_scores(pre_learning, sim) for sim in realignment_sims])
    true_scores    = compute_change_scores(pre_learning, post_learning)
    return subsel_scores, realign_scores, true_scores


# ─── Scoring helpers ──────────────────────────────────────────────────────────

def compute_statistics(true_score, simulated_distribution):
    z  = (true_score - np.mean(simulated_distribution)) / np.std(simulated_distribution)
    pt = scipy.stats.norm.sf(abs(z)) * 2
    return z, pt


def compute_mse(true_score, simulated_distribution):
    return np.mean((np.repeat(true_score, len(simulated_distribution)) - simulated_distribution) ** 2)


def compute_density(value, distribution):
    kde = scipy.stats.gaussian_kde(distribution, bw_method='scott')
    return kde(value)[0]


def compute_normalized_density(value, distribution, minn=None):
    kde   = scipy.stats.gaussian_kde(distribution, bw_method='scott')
    d_obs = kde(value)[0]
    d_max = np.max(kde(distribution))
    d_min = np.min(kde(distribution)) if minn is None else minn
    return (d_obs - d_min) / (d_max - d_min)


def compute_percentile(value, distribution):
    percentile = stats.percentileofscore(distribution, value) + 1e-6
    return np.abs(percentile - 50) / 50


# ─── Session-info helper ──────────────────────────────────────────────────────

def get_session_info(subject_info):
    im_session  = int(subject_info['im_session'].item())
    wmp_session = int(subject_info['wmp_session'].item())
    omp_session = int(subject_info['omp_session'].item())
    return im_session, wmp_session, omp_session


# ─── Compute summary_df ───────────────────────────────────────────────────────

def run_all_simulations(n_sims=1000, verbose=True):
    cumulative_info = helper.load_info_file()
    sim_dfs, summary_dfs = [], []

    for prop in PROPORTIONS:
        for sub in SUBJECTS:
            subject_info = cumulative_info[cumulative_info['subject_ID'] == sub]
            im_session, wmp_session, _ = get_session_info(subject_info)

            for ses_type, ses_num in zip(['IM', 'WMP'], [im_session, wmp_session]):
                t0 = time.time()
                first_run = 1 if ses_type == 'IM' else 2
                final_run = 4

                x0 = helper.get_realtime_outdata(sub, f'ses_0{ses_num}', first_run, data_type='decoded_angles')
                x1 = helper.get_realtime_outdata(sub, f'ses_0{ses_num}', final_run, data_type='decoded_angles')
                x0, x1 = x0[x0 == x0], x1[x1 == x1]

                subselection_sims, realignment_sims = run_simulations(x0, proportion=prop, n_sims=n_sims)
                subsel_scores, realign_scores, true_scores = score_simulations(
                    subselection_sims, realignment_sims, x0, x1)

                # per-simulation detail rows
                temp = pd.DataFrame({
                    'subselection_simulation_delta_variance': subsel_scores[:, 0],
                    'realignment_simulation_delta_variance':  realign_scores[:, 0],
                    'subselection_simulation_delta_entropy':  subsel_scores[:, 1],
                    'realignment_simulation_delta_entropy':   realign_scores[:, 1],
                })
                temp['subject_id']   = sub
                temp['session_type'] = ses_type
                temp['proportion']   = prop
                sim_dfs.append(temp)

                # summary rows
                names         = ['subselection', 'realignment']
                metrics       = ['variance', 'entropy']
                distributions = [subsel_scores, realign_scores]
                summary_rows  = {
                    'subject_id': [sub] * 4, 'session_type': [ses_type] * 4,
                    'proportion': [prop] * 4, 'MSE': [], 'true_score': [],
                    'instantaneous_density': [], 'proportional_density': [],
                    'max_normed_density': [], 'z': [], 'pval_2t': [],
                    'metric': [], 'simulated_change': [], 'percentile': [],
                    'absolute_z': [],
                }
                for idx0, idx1 in itertools.product([0, 1], [0, 1]):
                    these_scores = distributions[idx0][:, idx1]
                    true_score   = true_scores[idx1]
                    z, pv = compute_statistics(true_score, these_scores)
                    summary_rows['MSE'].append(compute_mse(true_score, these_scores))
                    summary_rows['true_score'].append(true_score)
                    summary_rows['instantaneous_density'].append(compute_density(true_score, these_scores))
                    summary_rows['proportional_density'].append(compute_normalized_density(true_score, these_scores))
                    summary_rows['z'].append(z)
                    summary_rows['pval_2t'].append(pv)
                    summary_rows['metric'].append(metrics[idx1])
                    summary_rows['simulated_change'].append(names[idx0])
                    summary_rows['percentile'].append(compute_percentile(true_score, these_scores))
                    summary_rows['absolute_z'].append(np.abs(z))
                    summary_rows['max_normed_density'].append(
                        compute_normalized_density(true_score, these_scores, minn=0))
                summary_dfs.append(pd.DataFrame(summary_rows))

                if verbose:
                    print(f'finished {sub}, {ses_type}, prop={prop} in {time.time()-t0:.2f}s')

    summary_df = pd.concat(summary_dfs).reset_index(drop=True)
    sim_df     = pd.concat(sim_dfs).reset_index(drop=True)

    os.makedirs(FINAL_RESULTS_PATH, exist_ok=True)
    os.makedirs(INTERMEDIATE_RESULTS_PATH, exist_ok=True)
    summary_df.to_csv(SUMMARY_FN)
    sim_df.to_csv(SIM_FN)
    print(f'Saved: {SUMMARY_FN}')
    print(f'Saved: {SIM_FN}')
    return summary_df


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_normalized_density(summary_df):
    colors      = sns.color_palette('Set2', 3)
    proportions = summary_df['proportion'].unique()
    fig, axes   = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)

    for ax, ses in zip(axes, ['IM', 'WMP']):
        df_temp = (summary_df[(summary_df['metric'] == 'variance') &
                              (summary_df['session_type'] == ses)]
                   .reset_index(drop=True))

        sns.barplot(data=df_temp, x='proportion', y='max_normed_density',
                    hue='simulated_change', hue_order=['subselection', 'realignment'],
                    palette=colors[1:], alpha=0.5, edgecolor='black', ax=ax, order=proportions)
        sns.stripplot(data=df_temp, x='proportion', y='max_normed_density',
                      hue='simulated_change', hue_order=['subselection', 'realignment'],
                      palette=colors[1:], alpha=0.9, edgecolor='k', linewidth=0.5,
                      order=proportions, ax=ax, jitter=False, size=3, dodge=True)

        for i, proportion in enumerate(proportions):
            pair_df = df_temp[df_temp['proportion'] == proportion]

            for sub in df_temp['subject_id'].unique():
                sub_df = pair_df[pair_df['subject_id'] == sub]
                if len(sub_df) == 2:
                    ax.plot([i - 0.2, i + 0.2], sub_df['max_normed_density'],
                            color='gray', alpha=0.5, linewidth=0.4)

            group_realign = pair_df[pair_df['simulated_change'] == 'realignment']['max_normed_density']
            group_subsel  = pair_df[pair_df['simulated_change'] == 'subselection']['max_normed_density']

            if len(group_realign) > 1 and len(group_subsel) > 1:
                _, p, _ = helper.permutation_test(
                    np.array([group_realign, group_subsel]),
                    n_iterations=NPERM, alternative='two-sided')
                meann, lowerr, upperr, _ = helper.bootstrap_ci(
                    np.array([group_realign, group_subsel]), n_boot=NPERM, verbose=0)
                d = helper.cohens_d_paired(np.array([group_realign, group_subsel]), verbose=0)
                print(f'{ses}, proportion={proportion}: p={p:.4f}, mean diff={meann:.2f}, '
                      f'95% CI=({lowerr:.2f}, {upperr:.2f}), d={d:.2f}')

                ax.text(i, 1.1, determine_symbol(p), ha='center', va='top', fontsize=10)
                ax.plot([i - 0.2, i + 0.2], [1.05, 1.05], color='black', linewidth=1)

        ax.axhline(0, linestyle='--', linewidth=1, c='k')
        ax.set(xlabel='Proportion', ylabel='Max normalized density',
               title=f'{ses} session', ylim=[-0.1, 1.1])
        ax.legend_.remove()

    fig.tight_layout()
    sns.despine()
    os.makedirs(PLOTS_PATH, exist_ok=True)
    out_fn = os.path.join(PLOTS_PATH, 'simulations_normalized_density.pdf')
    plt.savefig(out_fn, bbox_inches='tight', format='pdf', transparent=True)
    print(f'Saved: {out_fn}')
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_sims', type=int, default=1000,
                        help='Number of simulated distributions per condition (default: 1000)')
    parser.add_argument('-v', '--verbose', type=int, default=1)
    args = parser.parse_args()

    if os.path.exists(SUMMARY_FN):
        print(f'Loading existing summary from {SUMMARY_FN}')
        summary_df = pd.read_csv(SUMMARY_FN, index_col=0)
    else:
        print(f'Summary not found — running simulations (n_sims={args.n_sims}) ...')
        summary_df = run_all_simulations(n_sims=args.n_sims, verbose=bool(args.verbose))

    plot_normalized_density(summary_df)


if __name__ == '__main__':
    main()
