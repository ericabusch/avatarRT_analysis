# mask_size_results.py
# Shows per-subject variability in neurofeedback mask size and tests whether
# mask size predicts BCI or neural learning, as in the "mask size analysis"
# section of revision.ipynb.
#
# Output plots (in results/plots/):
#   - mask_size.pdf                  : per-subject voxel count barplot
#   - mask_size_learning.pdf         : Spearman rho (mask size ~ delta_BC / delta_EVR)

import numpy as np
import pandas as pd
import os
import nibabel as nib
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from config import *

MAIN_RESULTS   = os.path.join(FINAL_RESULTS_PATH, 'main_results.csv')

os.makedirs(PLOTS_PATH, exist_ok=True)
sns.set_context(context_params)

SUBJECTS = [s for s in SUB_IDS if s not in ['avatarRT_sub_09', 'avatarRT_sub_20']]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def bootstrap_spearman_ci(x, y, n_bootstrap=NBOOT, ci=95, random_state=SEED):
    n = len(x)
    rng = np.random.default_rng(random_state)
    boot_corrs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        boot_corrs[i] = spearmanr(x[idx], y[idx])[0]
    lower = np.percentile(boot_corrs, (100 - ci) / 2)
    upper = np.percentile(boot_corrs, 100 - (100 - ci) / 2)
    return np.mean(boot_corrs), (lower, upper), spearmanr(x, y)[1]


def plot_rhos_with_cis(rhos, lowers, uppers, labels=None, color='C0', ax=None, ylabel='Correlation (rho)'):
    rhos   = np.asarray(rhos)
    lowers = np.asarray(lowers)
    uppers = np.asarray(uppers)
    x = np.arange(len(rhos))
    if ax is None:
        _, ax = plt.subplots()
    yerr = np.vstack([rhos - lowers, uppers - rhos])
    ax.bar(x, rhos, yerr=yerr, color=color, capsize=0, lw=2, ec='k')
    if labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.axhline(0, ls='--', c='k')
    ax.set_ylim((-1, 1))
    return ax


# ─── Load data ────────────────────────────────────────────────────────────────

def load_data():
    """
    Load main_results.csv and replace n_voxels with values computed directly
    from each subject's mask.nii.gz.
    """
    main_df = pd.read_csv(MAIN_RESULTS, index_col=0).drop_duplicates()
    main_df = main_df[main_df['subject_id'].isin(SUBJECTS)]

    # Compute true n_voxels from mask files
    nvox = {}
    for sub in SUBJECTS:
        mask_fn = os.path.join(DATA_PATH, sub, 'reference', 'mask.nii.gz')
        if not os.path.exists(mask_fn):
            print(f'  WARNING: mask not found for {sub}')
            nvox[sub] = np.nan
        else:
            nvox[sub] = int(np.sum(nib.load(mask_fn).get_fdata()))

    main_df['n_voxels'] = main_df['subject_id'].map(nvox)

    # Wide format: one row per subject × session_type
    wide = (main_df
            .pivot_table(index=['subject_id', 'session_type', 'n_voxels'],
                         columns='metric', values='score')
            .reset_index())
    wide.columns.name = None
    return wide


# ─── Plot 1: per-subject voxel count ──────────────────────────────────────────

def plot_mask_size(wide):
    sub_vox = (wide[['subject_id', 'n_voxels']]
               .drop_duplicates()
               .sort_values('n_voxels'))
    sub_labels = [s.split('_')[-1] for s in sub_vox['subject_id']]
    pal = sns.color_palette('YlGnBu', n_colors=len(sub_vox))

    _, ax = plt.subplots(figsize=(7, 3))
    sns.barplot(data=sub_vox, x='subject_id', y='n_voxels', order=sub_vox['subject_id'].tolist(),
                palette=pal, edgecolor='k', linewidth=1, ax=ax)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xticklabels(sub_labels, rotation=45, ha='right', fontsize=9)
    ax.set(ylabel='Count', xlabel='Participant ID',
           title='Voxels included in neurofeedback mask')
    sns.despine()
    plt.tight_layout()
    out_fn = os.path.join(PLOTS_PATH, 'mask_size.pdf')
    plt.savefig(out_fn, format='pdf', transparent=True)
    print(f'Saved plot: {out_fn}')
    plt.show()


# ─── Plot 2: mask size ~ learning (BCI + neural) ──────────────────────────────

def plot_mask_correlations(wide):
    conds = ['IM', 'WMP', 'OMP']
    colors = [colors_main[c] for c in conds]

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
    rows = []

    for ax, metric, title, ylabel in [
        (axes[0], 'delta_BC',  'Mask size ~ BCI learning',    r'Spearman $\rho$'),
        (axes[1], 'delta_EVR', 'Mask size ~ neural learning', ''),
    ]:
        rhos, lowers, uppers = [], [], []
        print(f'\n{title}')
        for cond in conds:
            sub_df = wide[wide['session_type'] == cond].dropna(subset=['n_voxels', metric])
            rho, (ci_low, ci_high), p = bootstrap_spearman_ci(
                sub_df['n_voxels'].values, sub_df[metric].values, n_bootstrap=NBOOT)
            print(f'  {cond}: rho={rho:.3f}, 95% CI=[{ci_low:.3f}, {ci_high:.3f}], p={p:.3f}')
            rhos.append(rho)
            lowers.append(ci_low)
            uppers.append(ci_high)
            rows.append({
                'outcome':       metric,
                'session_type':  cond,
                'spearman_rho':  rho,
                'CI_lower':      ci_low,
                'CI_upper':      ci_high,
                'p_value':       p,
            })

        plot_rhos_with_cis(rhos, lowers, uppers, labels=conds,
                           color=colors, ax=ax, ylabel=ylabel)
        ax.set_title(title)

    sns.despine()
    fig.tight_layout()
    out_fn = os.path.join(PLOTS_PATH, 'mask_size_learning.pdf')
    plt.savefig(out_fn, format='pdf', transparent=True)
    print(f'Saved plot: {out_fn}')
    plt.show()

    return pd.DataFrame(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    wide = load_data()
    vox_summary = wide[['subject_id', 'n_voxels']].drop_duplicates().sort_values('n_voxels')
    print(f'n_voxels range: {vox_summary["n_voxels"].min():.0f} – {vox_summary["n_voxels"].max():.0f}')
    print(vox_summary.to_string(index=False))

    vox_fn = os.path.join(INTERMEDIATE_RESULTS_PATH, 'mask_size_per_subject.csv')
    vox_summary.to_csv(vox_fn)
    print(f'Saved: {vox_fn}')

    plot_mask_size(wide)
    corr_df = plot_mask_correlations(wide)

    corr_fn = os.path.join(INTERMEDIATE_RESULTS_PATH, 'mask_size_correlations.csv')
    corr_df.to_csv(corr_fn, index=False)
    print(f'Saved: {corr_fn}')


if __name__ == '__main__':
    main()
