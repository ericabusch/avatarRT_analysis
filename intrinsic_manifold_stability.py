# intrinsic_manifold_stability.py
# Analyzes stability of the intrinsic neural manifold across days using Gromov-Wasserstein distances.
# Based on the "Comparison of intrinsic manifold across days" section of revision.ipynb.
#
# Output files (in results/results_public/):
#   - revision_gromov_wasserstein.csv      : pairwise within-subject GW distances
#   - gromov_wasserstein_analysis_results.csv : z-scored within vs. between-subject distances

import numpy as np
import pandas as pd
import os, sys
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations
from scipy.spatial.distance import pdist, squareform
import analysis_helpers as helper
from plotting_functions import determine_symbol
from config import *

RESULTS_PUBLIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'results_public')
PLOTS_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'plots')
SCRATCH_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'scratch')
BTWN_SUBJ_FN   = os.path.join(RESULTS_PUBLIC, 'btwn_subj_manifold_gwd2.csv')
WITHIN_FN      = os.path.join(RESULTS_PUBLIC, 'revision_gromov_wasserstein.csv')
RESULTS_FN     = os.path.join(RESULTS_PUBLIC, 'gromov_wasserstein_analysis_results.csv')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.join(SCRATCH_DIR, 'joystick_analyses'), exist_ok=True)

shift_by = SHIFTBY  # = 2
SUBJECTS = [s for s in SUB_IDS if s not in ['avatarRT_sub_09', 'avatarRT_sub_20']]

sns.set_context(context_params)


# ─── T-PHATE manifold helpers ────────────────────────────────────────────────

def prep_joystick_data(subject_id):
    """Load (or compute) 20D T-PHATE embedding of joystick session data."""
    import nibabel as nib
    temp_fn = os.path.join(SCRATCH_DIR, 'joystick_analyses',
                           f'{subject_id}_20d_TPHATE_embedding.npy')
    if os.path.exists(temp_fn):
        print(f'  loading cached joystick embedding for {subject_id}')
        return np.load(temp_fn)
    print(f'  computing joystick embedding for {subject_id}...')
    mask_arr = nib.load(f'{DATA_PATH}/{subject_id}/reference/mask.nii.gz').get_fdata()
    data = []
    for run in range(1, 5):
        ds, ti, xi, zi = helper.load_all_joystick_data(subject_id, run, mask_arr, shift_by)
        data.append(ds[ti >= 0])
    voxel_data = np.concatenate(data, axis=0)
    tph_dst = helper.embed_tphate(voxel_data, n_components=20)
    np.save(temp_fn, tph_dst)
    return tph_dst

################ NOTE TO SELF  : CONTINUE MAKING SURE THAT ALL CODE USES THE ANALYSIS HELPERS VERSIONS OF FUNCTIONS FOR CONSISTENCY ################


def prep_rt_data(subject_id, session_id):
    """Load (or compute) 20D T-PHATE embedding of real-time session data."""
    temp_fn = os.path.join(SCRATCH_DIR, 'joystick_analyses',
                           f'{subject_id}_{session_id}_RT_20d_TPHATE_embedding.npy')
    if os.path.exists(temp_fn):
        print(f'  loading cached RT embedding: {temp_fn}')
        return np.load(temp_fn)
    print(f'  computing RT embedding for {subject_id} {session_id}...')
    data = []
    for run in range(1, 5):
        d = helper.get_realtime_data_preprocesssed(subject_id, session_id, run, data_type=None)
        data.append(d)
    voxel_data = np.concatenate(data, axis=0)
    tph_dst = helper.embed_tphate(voxel_data, n_components=20)
    np.save(temp_fn, tph_dst)
    return tph_dst


# ─── Analysis ─────────────────────────────────────────────────────────────────

def run_within_subject_gw(cumulative_info):
    """
    Compute pairwise GW distances within each subject between:
      joystick ↔ IM, WMP, OMP sessions, and WMP ↔ IM, WMP ↔ OMP.
    Saves revision_gromov_wasserstein.csv.
    """
    import ot
    df = pd.DataFrame(columns=['subject_id', 'sesA', 'sesB', 'gw2'])
    for subject_id in SUBJECTS:
        print(f'Within-subject GW: {subject_id}')
        subject_info = cumulative_info[cumulative_info['subject_ID'] == subject_id]
        im_session  = int(subject_info['im_session'].item())
        wmp_session = int(subject_info['wmp_session'].item())
        omp_session = int(subject_info['omp_session'].item())

        joystick_mani    = prep_joystick_data(subject_id)
        im_session_mani  = prep_rt_data(subject_id, f'ses_0{im_session}')
        wmp_session_mani = prep_rt_data(subject_id, f'ses_0{wmp_session}')
        omp_session_mani = prep_rt_data(subject_id, f'ses_0{omp_session}')

        JM  = squareform(pdist(joystick_mani,  'euclidean'))
        IMM = squareform(pdist(im_session_mani, 'euclidean'))
        WMM = squareform(pdist(wmp_session_mani,'euclidean'))
        OMM = squareform(pdist(omp_session_mani,'euclidean'))

        g0 = ot.gromov.gromov_wasserstein2(JM,  IMM)
        g1 = ot.gromov.gromov_wasserstein2(JM,  WMM)
        g2 = ot.gromov.gromov_wasserstein2(JM,  OMM)
        g3 = ot.gromov.gromov_wasserstein2(WMM, IMM)
        g4 = ot.gromov.gromov_wasserstein2(WMM, OMM)

        for sesA, sesB, gw in [('joystick','IM',g0),('joystick','WMP',g1),
                                ('joystick','OMP',g2),('WMP','IM',g3),('WMP','OMP',g4)]:
            df.loc[len(df)] = {'subject_id': subject_id, 'sesA': sesA, 'sesB': sesB, 'gw2': gw}

        print(f'  done {subject_id}')

    df.to_csv(WITHIN_FN)
    print(f'Saved: {WITHIN_FN}')
    return df


def run_between_subject_gw():
    """
    Compute pairwise GW distances between each pair of subjects comparing their
    joystick manifolds, and comparing sub0's joystick with sub1's RT sessions.
    Saves btwn_subj_manifold_gwd2.csv.
    """
    import ot
    btwn_df = pd.DataFrame(columns=['subject0', 'subject1',
                                    'dist_intrinsic2intrinsic', 'dist_intrinsic2other'])
    for (sub0, sub1) in permutations(SUBJECTS, 2):
        print(f'Between-subject GW: {sub0} vs {sub1}')
        joy0 = squareform(pdist(prep_joystick_data(sub0)))
        joy1 = squareform(pdist(prep_joystick_data(sub1)))

        sub1ses2 = squareform(pdist(prep_rt_data(sub1, 'ses_02')))
        sub1ses3 = squareform(pdist(prep_rt_data(sub1, 'ses_03')))
        sub1ses4 = squareform(pdist(prep_rt_data(sub1, 'ses_04')))

        g0 = ot.gromov.gromov_wasserstein2(joy0, joy1)
        g2 = ot.gromov.gromov_wasserstein2(joy0, sub1ses2)
        g3 = ot.gromov.gromov_wasserstein2(joy0, sub1ses3)
        g4 = ot.gromov.gromov_wasserstein2(joy0, sub1ses4)
        btwn_df.loc[len(btwn_df)] = {
            'subject0': sub0, 'subject1': sub1,
            'dist_intrinsic2intrinsic': g0,
            'dist_intrinsic2other': np.mean([g2, g3, g4])
        }

    btwn_df.to_csv(BTWN_SUBJ_FN)
    print(f'Saved: {BTWN_SUBJ_FN}')
    return btwn_df


def compute_zscored_results(df, btwn_subj_df):
    """
    For each subject: z-score their within-subject manifold distance relative to
    the between-subject distribution.
    Saves gromov_wasserstein_analysis_results.csv.
    """
    rnd_df = pd.DataFrame(columns=['subject_id',
                                   'self_dist_intrinsic2IMses_within',
                                   'self_dist_intrinsic2others_within',
                                   'zscore_dist_intrinsic2intrinsic_btwn',
                                   'zscore_dist_intrinsic2others_btwn'])
    tall_df = pd.DataFrame(columns=['subject_id', 'comparison', 'zscore'])

    for sub in SUBJECTS:
        self_df = df[(df['subject_id'] == sub) & (df['intrinsic_anchor'] == 1)]
        othr_df = btwn_subj_df[btwn_subj_df['subject0'] == sub]

        self_im     = self_df[self_df['sesB'] == 'IM']['gw2'].item()
        self_others = np.min(self_df['gw2'].values)

        intrin2intrin = np.random.choice(
            np.concatenate([[self_others], othr_df['dist_intrinsic2intrinsic'].values]), 10000)
        intrin2other  = np.random.choice(
            np.concatenate([[self_others], othr_df['dist_intrinsic2other'].values]), 10000)

        zs0 = (self_others - np.mean(intrin2intrin)) / np.std(intrin2intrin)
        zs1 = (self_others - np.mean(intrin2other))  / np.std(intrin2other)

        rnd_df.loc[len(rnd_df)] = {
            'subject_id': sub,
            'self_dist_intrinsic2IMses_within': self_im,
            'self_dist_intrinsic2others_within': self_others,
            'zscore_dist_intrinsic2intrinsic_btwn': zs0,
            'zscore_dist_intrinsic2others_btwn': zs1
        }
        tall_df.loc[len(tall_df)] = {'subject_id': sub, 'comparison': 'intrinsic2intrinsic', 'zscore': zs0}
        tall_df.loc[len(tall_df)] = {'subject_id': sub, 'comparison': 'intrinsic2others',    'zscore': zs1}

    rnd_df.to_csv(RESULTS_FN)
    print(f'Saved: {RESULTS_FN}')
    return rnd_df, tall_df


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_manifold_stability(rnd_df):
    """
    Two-panel barplot of z-scored within- vs. between-subject manifold distances.
    Left panel:  zscore_dist_intrinsic2others_btwn
    Right panel: zscore_dist_intrinsic2intrinsic_btwn
    Significance from one-sample permutation test vs. 0 (two-sided).
    """
    c = '#FF9C81'

    a = np.array([rnd_df['zscore_dist_intrinsic2intrinsic_btwn'].values, np.zeros(len(rnd_df))])
    b = np.array([rnd_df['zscore_dist_intrinsic2others_btwn'].values,    np.zeros(len(rnd_df))])

    _, p0, _ = helper.permutation_test(a, 10000, alternative='two-sided')
    _, p1, _ = helper.permutation_test(b, 10000, alternative='two-sided')
    # get confidence intervals for the means
    mean_a, lower_a, upper_a, _ = helper.bootstrap_ci(a[0], n_boot=10000, verbose=0)
    mean_b, lower_b, upper_b, _ = helper.bootstrap_ci(b[0], n_boot=10000, verbose=0)
    d_a = helper.cohens_d_paired(a[0], verbose=0)
    d_b = helper.cohens_d_paired(b[0], verbose=0)
    # display results with p-values and confidence intervals
    print(f'\nManifold stability — bootstrap CIs:')
    print(f'  intrinsic2intrinsic: mean={mean_a:.4f}  95%CI=[{lower_a:.4f}, {upper_a:.4f}]  d={d_a:.4f}')
    print(f'  intrinsic2others:    mean={mean_b:.4f}  95%CI=[{lower_b:.4f}, {upper_b:.4f}]  d={d_b:.4f}')

    print(f'\nManifold stability — permutation tests (two-sided, vs. 0):')
    print(f'  intrinsic2intrinsic: p = {p0:.4f}  {determine_symbol(p0)}')
    print(f'  intrinsic2others:    p = {p1:.4f}  {determine_symbol(p1)}')

    stats_rows = [
        {'comparison': 'intrinsic2others vs 0', 'test': 'permutation_test (n_iter=10000, alternative=two-sided)',
         'group1': 'intrinsic2others', 'group2': '0 (null)', 'n1': len(b[0]), 'n2': np.nan,
         'mean1': mean_b, 'mean2': 0, 'p_value': p1, 'ci_lower': lower_b, 'ci_upper': upper_b, 'cohens_d': d_b},
        {'comparison': 'intrinsic2intrinsic vs 0', 'test': 'permutation_test (n_iter=10000, alternative=two-sided)',
         'group1': 'intrinsic2intrinsic', 'group2': '0 (null)', 'n1': len(a[0]), 'n2': np.nan,
         'mean1': mean_a, 'mean2': 0, 'p_value': p0, 'ci_lower': lower_a, 'ci_upper': upper_a, 'cohens_d': d_a},
    ]
    stats_df = pd.DataFrame(stats_rows)
    stats_df['significant_0.05'] = stats_df['p_value'] < 0.05
    stats_fn = os.path.join(RESULTS_PUBLIC, 'manifold_stability_stats.csv')
    stats_df.to_csv(stats_fn, index=False)
    print(f'Saved statistics to {stats_fn}')

    fig, ax = plt.subplots(1, 2, figsize=(3, 4), sharey=False)

    # Left panel: intrinsic2others
    sns.barplot(y=b[0], ax=ax[0], color=c, edgecolor='k', linewidth=2, alpha=1)
    g = sns.stripplot(y=b[0], ax=ax[0], color=c, edgecolor='k', linewidth=1, alpha=1)
    g.set(ylabel='z-score distance')
    ax[0].axhline(0, color='k', ls='--')
    ax[0].set(ylim=[-2.5, 1.1])
    ax[0].set_title('joystick vs.\nRT sessions', fontsize=9)
    ax[0].text(x=-0.1, y=0.65, s=determine_symbol(p0), size=12)

    # Right panel: intrinsic2intrinsic
    sns.barplot(y=a[0], ax=ax[1], color=c, edgecolor='k', linewidth=2, alpha=1)
    g = sns.stripplot(y=a[0], ax=ax[1], color=c, edgecolor='k', linewidth=1, alpha=1)
    g.set(ylim=[-2.5, 1.1])
    ax[1].axhline(0, color='k', ls='--')
    ax[1].set_title('joystick vs.\njoystick (others)', fontsize=9)
    ax[1].text(x=-0.1, y=0.65, s=determine_symbol(p1), size=12)

    sns.despine()
    fig.tight_layout()

    out_fn = os.path.join(PLOTS_DIR, 'manifold_stability.pdf')
    plt.savefig(out_fn, format='pdf', transparent=True)
    print(f'Saved plot: {out_fn}')
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    both_exist = os.path.exists(WITHIN_FN) and os.path.exists(RESULTS_FN)

    if both_exist:
        print('Result files found — loading and plotting.')
        df      = pd.read_csv(WITHIN_FN, index_col=0)
        rnd_df  = pd.read_csv(RESULTS_FN, index_col=0)
    else:
        print('Result files not found — running analysis.')
        cumulative_info = helper.load_info_file()

        # Step 1: within-subject pairwise GW distances
        if os.path.exists(WITHIN_FN):
            print(f'Loading existing: {WITHIN_FN}')
            df = pd.read_csv(WITHIN_FN, index_col=0)
        else:
            df = run_within_subject_gw(cumulative_info)

        df['intrinsic_anchor'] = (df['sesA'] == 'joystick').astype(int)

        # Step 2: between-subject GW distances
        if os.path.exists(BTWN_SUBJ_FN):
            print(f'Loading existing: {BTWN_SUBJ_FN}')
            btwn_subj_df = pd.read_csv(BTWN_SUBJ_FN, index_col=0)
        else:
            btwn_subj_df = run_between_subject_gw()

        # Step 3: z-score within vs between
        rnd_df, _ = compute_zscored_results(df, btwn_subj_df)

    # Plot
    plot_manifold_stability(rnd_df)


if __name__ == '__main__':
    main()
