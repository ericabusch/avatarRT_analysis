# eigenspectrum_results.py
# Analyzes and visualizes the control-space eigenspectrum for each subject.
# Based on the "Run eigenspectrum for each subject" section of revision.ipynb.
#
# Output file (in results/results_public/):
#   - manifold_eigenspectrum.csv  : % variance explained per component per subject

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from config import *

RESULTS_PUBLIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'results_public')
PLOTS_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'plots')
EIGENSPECTRUM_FN = os.path.join(RESULTS_PUBLIC, 'manifold_eigenspectrum.csv')
MAIN_RESULTS_FN  = os.path.join(RESULTS_PUBLIC, 'main_results.csv')

os.makedirs(PLOTS_DIR, exist_ok=True)

# Subjects for analysis (exclude scanner/sleep issues)
SUBJECTS = [s for s in SUB_IDS if s not in ['avatarRT_sub_09', 'avatarRT_sub_20']]

# Component indices (1-indexed) that correspond to trained NFB conditions
SPECIAL_IDX  = [1, 2, 20]   # IM, WMP, OMP
NFB_COLORS   = {'IM': colors_main['IM'], 'WMP': colors_main['WMP'],
                'OMP': colors_main['OMP'], 'untrained': color_gray}

sns.set_context(context_params)


# ─── Analysis ─────────────────────────────────────────────────────────────────

def compute_eigenspectra():
    """
    Load each subject's bottleneck.npy, compute % variance explained per component
    from the covariance eigenspectrum. Returns V: (n_subjects, n_components) array.
    """
    var_explaineds = []
    subjects_used  = []
    for sub in SUBJECTS:
        fn = f'{DATA_PATH}/{sub}/model/bottleneck.npy'
        if not os.path.exists(fn):
            print(f'  WARNING: missing {fn}')
            continue
        data = np.load(fn)
        cov  = np.cov(data, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        var_explained = eigvals / eigvals.sum() * 100
        var_explaineds.append(var_explained)
        subjects_used.append(sub)

    V = np.array(var_explaineds)
    return V, subjects_used


def build_eigenspectrum_df(V, subjects_used):
    """Convert V array to long-form DataFrame and save to CSV."""
    n_components = V.shape[1]
    rows = []
    for j, sub in enumerate(subjects_used):
        for i in range(n_components):
            if i == 0:
                cond = 'IM'
            elif i == 1:
                cond = 'WMP'
            elif i == n_components - 1:
                cond = 'OMP'
            else:
                cond = 'untrained'
            rows.append({'subject_id': sub, 'component_idx': i,
                         'PEV_initial': V[j, i], 'nfb_condition': cond})
    df = pd.DataFrame(rows)
    df.to_csv(EIGENSPECTRUM_FN)
    print(f'Saved: {EIGENSPECTRUM_FN}')
    return df


def load_or_compute():
    """Load eigenspectrum CSV if present, otherwise compute and save it."""
    if os.path.exists(EIGENSPECTRUM_FN):
        print('Eigenspectrum CSV found — loading.')
        df = pd.read_csv(EIGENSPECTRUM_FN, index_col=0)
        # Reconstruct V (n_subjects × n_components) from CSV
        subjects_used = df['subject_id'].unique().tolist()
        V = np.array([df[df['subject_id'] == s]['PEV_initial'].values for s in subjects_used])
        return df, V, subjects_used
    else:
        print('Eigenspectrum CSV not found — computing.')
        V, subjects_used = compute_eigenspectra()
        df = build_eigenspectrum_df(V, subjects_used)
        return df, V, subjects_used


# ─── Visualization helpers ────────────────────────────────────────────────────

def tiny_to_sci(x, precision=1):
    if np.isscalar(x):
        return f"{x:.{precision}e}"
    return [f"{val:.{precision}e}" for val in np.asarray(x)]


# ─── Plot 1: per-subject eigenspectrum grid ───────────────────────────────────

def plot_eigenspectrum_grid(V, subjects_used, nrows=3, ncols=6, figsize=(16, 10)):
    """
    Per-subject grid of % variance explained by component.
    NFB-condition components (IM=1, WMP=2, OMP=last) highlighted in red.
    """
    _, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()
    n_components = V.shape[1]
    special_1idx = SPECIAL_IDX  # 1-indexed

    for i, sub in enumerate(subjects_used):
        ax = axes[i]
        var_exp = V[i]
        x = np.arange(1, n_components + 1)
        sns.lineplot(x=x, y=var_exp, ax=ax, color='C0', marker='o', markersize=4)
        # Highlight NFB components
        y_special = [var_exp[s - 1] for s in special_1idx]
        sns.scatterplot(x=special_1idx, y=y_special,
                        color='red', marker='x', s=80, zorder=10, ax=ax)
        short_id = sub.split('_')[-1]
        ax.set_title(short_id, fontsize=12)
        if i % ncols == 0:
            ax.set_ylabel('% Variance Explained', fontsize=10)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel('Component', fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        sns.despine(ax=ax)

    for j in range(len(subjects_used), nrows * ncols):
        axes[j].axis('off')

    plt.tight_layout()
    out_fn = os.path.join(PLOTS_DIR, 'eigenspectrum_grid.pdf')
    plt.savefig(out_fn, format='pdf', transparent=True)
    print(f'Saved plot: {out_fn}')
    plt.show()


# ─── Plot 2: PEV by NFB condition (summary) ───────────────────────────────────

def plot_pev_by_condition(df):
    """
    Barplot + strip comparing % variance explained for each NFB condition
    (IM, WMP, OMP) vs. the untrained components.
    """
    # For untrained, take mean across untrained components per subject
    pev_df = df[df['nfb_condition'] != 'untrained'].copy()
    untrained_mean = (df[df['nfb_condition'] == 'untrained']
                      .groupby('subject_id')['PEV_initial'].mean()
                      .reset_index()
                      .assign(nfb_condition='untrained'))
    plot_df = pd.concat([pev_df, untrained_mean], ignore_index=True)

    cond_order  = ['IM', 'WMP', 'OMP', 'untrained']
    palette = [colors_main['IM'], colors_main['WMP'], colors_main['OMP'], color_gray]

    _, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(data=plot_df, x='nfb_condition', y='PEV_initial', hue='nfb_condition',
                order=cond_order, palette=dict(zip(cond_order, palette)),
                edgecolor='k', linewidth=1.5, ax=ax)
    sns.stripplot(data=plot_df, x='nfb_condition', y='PEV_initial', hue='nfb_condition',
                  order=cond_order, palette=dict(zip(cond_order, palette)),
                  edgecolor='k', linewidth=0.8, jitter=True, ax=ax)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlabel('NFB condition')
    ax.set_ylabel('% Variance Explained (initial)')
    ax.axhline(0, color='k', ls='--', linewidth=0.8)
    sns.despine()
    plt.tight_layout()
    out_fn = os.path.join(PLOTS_DIR, 'eigenspectrum_by_condition.pdf')
    plt.savefig(out_fn, format='pdf', transparent=True)
    print(f'Saved plot: {out_fn}')
    plt.show()


# ─── Plot 3: correlation with learning ────────────────────────────────────────

def plot_pev_correlations(df):
    """
    Merge eigenspectrum data with behavioral/neural learning outcomes and
    plot Spearman correlations:
      - delta_varexp_IM_WMP (PEV_IM - PEV_WMP) vs delta_BC_WMP
      - delta_varexp_IM_WMP vs delta_EVR_IM - delta_EVR_WMP
    """
    if not os.path.exists(MAIN_RESULTS_FN):
        print(f'  {MAIN_RESULTS_FN} not found — skipping correlation plots.')
        return

    # Build per-subject summary from eigenspectrum CSV
    piv = df.pivot_table(index='subject_id', columns='nfb_condition',
                         values='PEV_initial', aggfunc='first')
    # delta_varexp = PEV[IM component] - PEV[WMP component]
    if 'IM' not in piv.columns or 'WMP' not in piv.columns:
        print('  Cannot compute delta_varexp — missing IM or WMP in CSV.')
        return
    summary = pd.DataFrame({
        'subject_id': piv.index,
        'PEV_IM':  piv['IM'].values,
        'PEV_WMP': piv['WMP'].values,
        'delta_varexp_IM_WMP': piv['IM'].values - piv['WMP'].values,
    })

    # Load behavioral/neural outcomes
    main = pd.read_csv(MAIN_RESULTS_FN, index_col=0)
    wide = main.pivot_table(index=['subject_id', 'session_type'],
                            columns='metric', values='score').reset_index()
    bc_wmp  = wide[wide['session_type'] == 'WMP'][['subject_id', 'delta_BC']].rename(
                  columns={'delta_BC': 'delta_bc_WMP'})
    bc_im   = wide[wide['session_type'] == 'IM'][['subject_id', 'delta_BC']].rename(
                  columns={'delta_BC': 'delta_bc_IM'})
    evr_wmp = wide[wide['session_type'] == 'WMP'][['subject_id', 'delta_EVR']].rename(
                  columns={'delta_EVR': 'delta_evr_WMP'})
    evr_im  = wide[wide['session_type'] == 'IM'][['subject_id', 'delta_EVR']].rename(
                  columns={'delta_EVR': 'delta_evr_IM'})

    df1 = (summary
           .merge(bc_wmp,  on='subject_id', how='inner')
           .merge(bc_im,   on='subject_id', how='inner')
           .merge(evr_wmp, on='subject_id', how='inner')
           .merge(evr_im,  on='subject_id', how='inner'))
    df1['delta_evr_IM_WMP'] = df1['delta_evr_IM'] - df1['delta_evr_WMP']

    _, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1: per-subject bar of delta_varexp_IM_WMP
    df1['sub_label'] = [s.split('_')[-1] for s in df1.subject_id]
    pal = sns.color_palette('YlGnBu', n_colors=len(df1))
    g = sns.barplot(data=df1, x='sub_label', y='delta_varexp_IM_WMP',
                    hue='sub_label', palette=pal,
                    ax=axes[0], edgecolor='k', linewidth=1)
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    g.set(ylabel='Diff % var explained',
          title='PEV difference (IM − WMP component)')
    axes[0].tick_params(axis='x', rotation=45, labelsize=8)

    # Panel 2: delta_bc_WMP ~ delta_varexp_IM_WMP
    r, p = spearmanr(df1['delta_varexp_IM_WMP'], df1['delta_bc_WMP'])
    g = sns.regplot(data=df1, x='delta_varexp_IM_WMP', y='delta_bc_WMP', ax=axes[1])
    g.set(xlabel='PEV difference (IM − WMP component)',
          ylabel=r'$\Delta$ Brain Control (%), WMP session',
          title='')
    axes[1].text(0.96, 0.04, f'rho={r:.2f}\np={p:.3f}',
                 transform=axes[1].transAxes, fontsize=11,
                 va='bottom', ha='right',
                 bbox=dict(facecolor='white', alpha=0.8))

    # Panel 3: delta_evr_IM_WMP ~ delta_varexp_IM_WMP
    r, p = spearmanr(df1['delta_varexp_IM_WMP'], df1['delta_evr_IM_WMP'])
    g = sns.regplot(data=df1, x='delta_varexp_IM_WMP', y='delta_evr_IM_WMP', ax=axes[2])
    g.set(xlabel='PEV difference (IM − WMP component)',
          ylabel=r'Difference $\Delta$ EVR (IM − WMP)',
          title='')
    axes[2].text(0.96, 0.04, f'rho={r:.2f}\np={p:.3f}',
                 transform=axes[2].transAxes, fontsize=11,
                 va='bottom', ha='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    sns.despine()
    plt.tight_layout()
    out_fn = os.path.join(PLOTS_DIR, 'eigenspectrum_correlations.pdf')
    plt.savefig(out_fn, format='pdf', transparent=True)
    print(f'Saved plot: {out_fn}')
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    df, V, subjects_used = load_or_compute()
    print(f'\nLoaded eigenspectra: {V.shape[0]} subjects × {V.shape[1]} components')

    plot_eigenspectrum_grid(V, subjects_used)
    plot_pev_by_condition(df)
    plot_pev_correlations(df)


if __name__ == '__main__':
    main()
