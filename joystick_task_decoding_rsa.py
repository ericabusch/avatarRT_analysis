# joystick_results.py
# Loads joystick decoding results and plots them in the same style as
# the neural_results.ipynb notebook.  If the results file is absent,
# the script exits gracefully with a message.

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True
import matplotlib.pyplot as plt
import seaborn as sns
import analysis_helpers as helper
from plotting_functions import determine_symbol
from config import *

RESULTS_PUBLIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'results_public')
PLOTS_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'plots')
# Prefer new output from run_joystick_analyses.py; fall back to legacy file
_NEW_JOY_FILE  = os.path.join(INTERMEDIATE_RESULTS_PATH,
                               'joystick_figure1_himalaya_run_cross_validation.csv')
_OLD_JOY_FILE  = os.path.join(RESULTS_PUBLIC, 'joystick_decoding_results_final.csv')

os.makedirs(PLOTS_DIR, exist_ok=True)

if os.path.isfile(_NEW_JOY_FILE):
    _joy_fn, _new_format = _NEW_JOY_FILE, True
elif os.path.isfile(_OLD_JOY_FILE):
    _joy_fn, _new_format = _OLD_JOY_FILE, False
else:
    print('No joystick results file found.')
    print(f'Run run_joystick_analyses.py first to generate results.')
    raise SystemExit(0)

# ── Load and reshape ──────────────────────────────────────────────────────────

sns.set_context(context_params)

if _new_format:
    # New file: numeric index, embedding_type already a column
    raw = pd.read_csv(_joy_fn, index_col=0)
else:
    # Legacy file: embedding_type was the index
    raw = pd.read_csv(_joy_fn, index_col=0).reset_index()
    raw.columns = raw.columns.str.strip()
    raw = raw.rename(columns={raw.columns[0]: 'embedding_type'})

# Keep voxel vs T-PHATE, metrics MSE (himalaya) and mantel z-score
raw = raw[raw['embedding_type'].isin(['voxel', 'tphate']) &
          raw['metric'].isin(['himalaya', 'z', 'r'])]

# Keep only subjects present in both embedding types (matched pairs)
tphate_subs = set(raw[raw['embedding_type'] == 'tphate']['subject_id'])
voxel_subs  = set(raw[raw['embedding_type'] == 'voxel']['subject_id'])
shared_subs = sorted(tphate_subs & voxel_subs)
raw = raw[raw['subject_id'].isin(shared_subs)]

# Pivot to wide format: one row per subject, columns = embedding_type × metric
wide = raw.pivot_table(
    index='subject_id',
    columns=['embedding_type', 'metric'],
    values='score', aggfunc='mean',
).reset_index()
wide.columns = ['subject_id'] + ['_'.join(c) for c in wide.columns[1:]]

print(f'Subjects: {len(shared_subs)}  |  columns: {wide.columns.tolist()}')

# embedding=0 → voxel, embedding=1 → tphate
# Build a long-format df matching the notebook's joy_df structure

joy_rows = []
for _, row in wide.iterrows():
    joy_rows.append({'subject_id': row['subject_id'], 'embedding': 0,
                     'mse': row['voxel_himalaya'], 'mantel_z': row['voxel_z'],
                     'mantel_rho': row['voxel_r']})
    joy_rows.append({'subject_id': row['subject_id'], 'embedding': 1,
                     'mse': row['tphate_himalaya'], 'mantel_z': row['tphate_z'],
                     'mantel_rho': row['tphate_r']})
joy_df = pd.DataFrame(joy_rows)

# ── Plot:  ──────────────────────────────────────

pal      = [sns.color_palette('Set1', 6)[3]] * 2   # same color both bars (notebook style)
ynames   = ['mse', 'mantel_z']
ylabels  = ['Mean-squared error', 'z-score']
yplus    = [0.5, 2]          # y-offset for significance annotation

np.random.seed(SEED)

all_stats = []

for i, (yname, ylabel, yoff) in enumerate(zip(ynames, ylabels, yplus)):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))

    g = sns.barplot(x='embedding', y=yname, data=joy_df,
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

    # Statistics: paired permutation test + bootstrap CI + Cohen's d
    arr0, arr1 = np.array(points0), np.array(points1)
    _, pv, _ = helper.permutation_test(
        np.array([arr0, arr1]), n_iterations=10000, alternative='two-sided')
    m0, lo0, hi0, _       = helper.bootstrap_ci(arr0, n_boot=10000, verbose=0)
    m1, lo1, hi1, _       = helper.bootstrap_ci(arr1, n_boot=10000, verbose=0)
    m_diff, lo_d, hi_d, _ = helper.bootstrap_ci(arr0 - arr1, n_boot=10000, verbose=0)
    d_paired               = helper.cohens_d_paired(arr0 - arr1, verbose=0)
    print(f'{yname}  voxel:  mean={m0:.4f}  95%CI=[{lo0:.4f},{hi0:.4f}]')
    print(f'{yname}  T-PHATE:  mean={m1:.4f}  95%CI=[{lo1:.4f},{hi1:.4f}]')
    print(f'{yname}  voxel−T-PHATE:  mean={m_diff:.4f}  p={pv:.4f}  95%CI=[{lo_d:.4f},{hi_d:.4f}]  Cohen\'s d={d_paired:.4f}')
    pv_plot = pv
    if pv_plot > 0.5:
        pv_plot = 1 - pv_plot

    pstr = determine_symbol(pv_plot)
    all_stats.append({
        'comparison': f'{yname}: voxel vs T-PHATE',
        'test': 'permutation_test (n_iter=10000, alternative=two-sided)',
        'group1': 'voxel', 'group2': 'T-PHATE',
        'n1': len(arr0), 'n2': len(arr1),
        'mean1': m0, 'mean2': m1,
        'p_value': pv, 'ci_lower': lo_d, 'ci_upper': hi_d,
        'cohens_d': d_paired,
    })
    yloc = np.max(np.concatenate((points0, points1))) + yoff

    if pstr is not None:
        ax.axhline(y=yloc, xmin=0.25, xmax=0.75, color='k', lw=1)
        ax.text(x=0.5, y=yloc + (yoff * 0.005), s=pstr, ha='center')

    g.set(xticklabels=['voxel', 'T-PHATE'], ylabel=ylabel, xlabel='')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'joystick_{yname}.pdf'),
                transparent=True, bbox_inches='tight', format='pdf')
    # plt.show()
    print()

stats_df = pd.DataFrame(all_stats)
stats_df['significant_0.05'] = stats_df['p_value'] < 0.05
stats_fn = os.path.join(RESULTS_PUBLIC, 'joystick_stats.csv')
stats_df.to_csv(stats_fn, index=False)
print(f'Saved statistics to {stats_fn}')

print(f'\nPlots saved to {PLOTS_DIR}')
print('Joystick results complete.')
