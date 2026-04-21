# order_effects.py
# Analyzes the effect of counterbalancing order (WMP-first vs OMP-first) on
# delta_BC and delta_EVR, as in the "order effects" section of revision.ipynb.
#
# Outputs (in results/results_public/):
#   - order_effects_lm_results.csv : OLS F-tests and coefficients for both outcomes
# Outputs (in results/plots/):
#   - order_effects_barplot.pdf    : delta_BC by session type × counterbalancing order

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import analysis_helpers as helper
from plotting_functions import determine_symbol
from config import *

RESULTS_PUBLIC  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'results_public')
FINAL_RESULTS   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'final_results')
PLOTS_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'plots')
LM_RESULTS_FN   = os.path.join(RESULTS_PUBLIC, 'order_effects_lm_results.csv')
BEHAV_SESSION_FN = os.path.join(FINAL_RESULTS, 'behavioral_change_session.csv')
MAIN_RESULTS_FN  = os.path.join(RESULTS_PUBLIC, 'main_results.csv')

os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_context(context_params)

PALETTE = ["#fca5a5", "#ff0000"]   # light red = WMP-first, dark red = OMP-first


# ─── Helpers ──────────────────────────────────────────────────────────────────

# ─── Analysis ─────────────────────────────────────────────────────────────────

def load_data():
    """
    Returns a wide-format DataFrame with one row per subject × session_type,
    containing delta_BC, delta_EVR, and wmp_first.
    """
    # Behavioral delta_BC (all subjects, all sessions)
    behav = pd.read_csv(BEHAV_SESSION_FN, index_col=0)
    behav['wmp_first'] = behav['subject_id'].isin(WM_FIRST).astype(int)

    # delta_EVR from main_results (neural)
    if os.path.exists(MAIN_RESULTS_FN):
        main = pd.read_csv(MAIN_RESULTS_FN, index_col=0)
        evr = (main[main['metric'] == 'delta_EVR']
               [['subject_id', 'session_type', 'score']]
               .drop_duplicates(subset=['subject_id', 'session_type'])
               .rename(columns={'score': 'delta_EVR'}))
        df = behav.merge(evr, on=['subject_id', 'session_type'], how='left')
    else:
        print(f'  {MAIN_RESULTS_FN} not found — delta_EVR model will be skipped.')
        df = behav.copy()
        df['delta_EVR'] = np.nan

    # Exclude neural problem subjects from EVR analysis only
    excl = [f'avatarRT_sub_{n:02d}' for n in exclude_from_neural_analyses]
    df.loc[df['subject_id'].isin(excl), 'delta_EVR'] = np.nan

    return df


def run_order_models(df):
    """
    Fit OLS models for delta_BC and delta_EVR ~ session_type * wmp_first
    (WMP and OMP sessions only). Returns a results DataFrame.
    """
    rows = []
    sub_df = df[df['session_type'] != 'IM'].copy()

    for var in ['delta_BC', 'delta_EVR']:
        data = sub_df.dropna(subset=[var]).copy()
        if data.empty:
            continue

        result = ols(f'{var} ~ C(session_type) * C(wmp_first)', data=data).fit()
        anova  = anova_lm(result, typ=2)

        print(f'\n{"="*60}')
        print(f'RESULTS FOR {var}')
        print(f'{"="*60}')
        print(result.summary())
        print('\nType II ANOVA:')
        print(anova.round(4))

        # ── F-test rows ──────────────────────────────────────────────────────
        for term in ['C(session_type)', 'C(wmp_first)', 'C(session_type):C(wmp_first)']:
            if term not in anova.index:
                continue
            rows.append({
                'outcome':   var,
                'term':      term,
                'row_type':  'F_test',
                'df_num':    anova.loc[term, 'df'],
                'df_den':    anova.loc['Residual', 'df'],
                'F':         anova.loc[term, 'F'],
                'p_value':   anova.loc[term, 'PR(>F)'],
                'beta':      np.nan,
                'se':        np.nan,
                't':         np.nan,
            })

        # ── Coefficient rows ─────────────────────────────────────────────────
        for coef_name in result.params.index:
            rows.append({
                'outcome':   var,
                'term':      coef_name,
                'row_type':  'coefficient',
                'df_num':    np.nan,
                'df_den':    np.nan,
                'F':         np.nan,
                'p_value':   result.pvalues[coef_name],
                'beta':      result.params[coef_name],
                'se':        result.bse[coef_name],
                't':         result.tvalues[coef_name],
            })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(LM_RESULTS_FN)
    print(f'\nSaved: {LM_RESULTS_FN}')
    return results_df


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_order_barplot(df):
    """
    Bar graph of delta_BC split by session type (WMP, OMP) and counterbalancing
    order (wmp_first). Significance annotations for between-group and per-bar tests.
    """
    plot_df = df[df['session_type'] != 'IM'].copy()
    ses_order  = ['WMP', 'OMP']
    hue_order  = [1, 0]
    bar_width  = 0.35

    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.barplot(data=plot_df, x='session_type', y='delta_BC', hue='wmp_first',
                order=ses_order, hue_order=hue_order,
                palette=PALETTE, edgecolor='k', linewidth=1, alpha=0.8, ax=ax)
    sns.stripplot(data=plot_df, x='session_type', y='delta_BC', hue='wmp_first',
                  order=ses_order, hue_order=hue_order,
                  dodge=True, palette=PALETTE, edgecolor='k', linewidth=0.4,
                  jitter=True, legend=False, ax=ax)

    ax.axhline(0, ls='--', c='k', linewidth=0.8)
    ax.set(ylim=[-0.8, 1.2],
           yticks=np.arange(-0.8, 1.21, 0.4),
           yticklabels=[f'{int(v*100)}' for v in np.arange(-0.8, 1.21, 0.4)],
           ylabel=r'$\Delta$ Brain Control (%)',
           xlabel='Session type')

    # Significance annotations
    for j, ses in enumerate(ses_order):
        v0 = plot_df[(plot_df['session_type'] == ses) & (plot_df['wmp_first'] == 1)]['delta_BC']
        v1 = plot_df[(plot_df['session_type'] == ses) & (plot_df['wmp_first'] == 0)]['delta_BC']

        x0 = j - bar_width / 2
        x1 = j + bar_width / 2
        ymax = max(v0.max(), v1.max())

        # Between-group comparison (WMP-first > OMP-first)
        _, p_btwn = ttest_ind(v0, v1, permutations=10000, alternative='greater')
        ax.plot([x0, x1], [0.7, 0.7], lw=1, c='k')
        ax.text(j, ymax + 0.18, determine_symbol(p_btwn), ha='center', va='bottom',
                color='k', fontsize=12)

        # Per-bar one-sample tests vs 0
        _, p0, _ = helper.permutation_test(np.array([v0.values, np.zeros(len(v0))]), 10000, alternative='greater')
        _, p1, _ = helper.permutation_test(np.array([v1.values, np.zeros(len(v1))]), 10000, alternative='greater')
        _, lower0, upper0 = helper.bootstrap_ci(v0.values, n_boot=10000)
        _, lower1, upper1 = helper.bootstrap_ci(v1.values, n_boot=10000)
        print(f'{ses} WMP-first: p={p0:.4f}, 95% CI=[{lower0:.4f}, {upper0:.4f}]')
        print(f'{ses} OMP-first: p={p1:.4f}, 95% CI=[{lower1:.4f}, {upper1:.4f}]')
        ax.text(x0, ymax + 0.05, determine_symbol(p0), ha='center', va='bottom',
                color='k', fontsize=9)
        ax.text(x1, ymax + 0.05, determine_symbol(p1), ha='center', va='bottom',
                color='k', fontsize=9)

    # Fix legend
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ['WMP then OMP', 'OMP then WMP'],
              title='', fontsize=8, loc='lower right')

    sns.despine()
    plt.tight_layout()
    out_fn = os.path.join(PLOTS_DIR, 'order_effects_barplot.pdf')
    plt.savefig(out_fn, format='pdf', transparent=True)
    print(f'Saved plot: {out_fn}')
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    df = load_data()
    run_order_models(df)
    plot_order_barplot(df)


if __name__ == '__main__':
    main()
