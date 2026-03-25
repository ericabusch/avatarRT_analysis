# neural_EVR_results.py
# Replicates the neural EVR analysis from neural_EVR_analysis.py and
# neural_results.ipynb.  If main_results.csv does not exist, runs
# delta_evr_across_components for every subject, filters the congruent
# condition, merges with behavioural runwise delta_BC, and saves the
# combined file.  If main_results.csv already exists, loads it and
# produces plots and mixed-effects regressions.

import numpy as np
import pandas as pd
import os, sys
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import analysis_helpers as helper
from plotting_functions import make_barplot_points, determine_symbol
from neural_EVR_analysis import delta_evr_across_components
from config import *

RESULTS_PUBLIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'results_public')
PLOTS_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'plots')
MAIN_RESULTS   = os.path.join(RESULTS_PUBLIC, 'main_results.csv')

os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Build main_results.csv if it does not already exist ──────────────────────

if not os.path.isfile(MAIN_RESULTS):
    print('main_results.csv not found — running delta_evr_across_components ...')

    cumulative_info = helper.load_info_file()
    sub_list, sub_list_all = [], []

    for subject_id in SUB_IDS:
        subject_info = cumulative_info[cumulative_info['subject_ID'] == subject_id]

        im_session  = int(subject_info['im_session'].item())
        wmp_session = int(subject_info['wmp_session'].item())
        omp_session = int(subject_info['omp_session'].item())

        wmp_component = int(subject_info['wmp_component'].item())
        omp_component = int(subject_info['omp_component'].item())
        wmp_idx = 1 if wmp_component == 1 else wmp_component - 1
        omp_idx = omp_component - 1

        session_name_mapping = {
            'IM':  [im_session,  0,       0],
            'WMP': [wmp_session, wmp_idx, wmp_component],
            'OMP': [omp_session, omp_idx, omp_component],
        }

        try:
            results, subtraction_results = delta_evr_across_components(
                subject_id, session_name_mapping)
        except Exception as e:
            print(f'  {subject_id}: failed ({e})')
            continue

        subtraction_results['subject_id'] = subject_id
        sub_list.append(subtraction_results)
        results['subject_id'] = subject_id
        sub_list_all.append(results)
        print(f'  {subject_id}: done')

    sub_overall = pd.concat(sub_list).reset_index(drop=True)

    # Keep only the congruent condition (NFB component for that session)
    cong_df = sub_overall[sub_overall['congruent'] == True].copy()
    cong_df = cong_df.rename(columns={'delta_run_perturb': 'delta_EVR',
                                       'session_number':    'session_number'})

    # Load pre-computed behavioural runwise delta_BC (observed subjects only)
    behav = pd.read_csv(
        os.path.join(RESULTS_PUBLIC, 'behavioral_change_runwise_with_simulations.csv'),
        index_col=0)
    behav = behav[behav['simulated'] == 0][['subject_id', 'session_type', 'delta_BC']].copy()

    # Annotate order flag
    cong_df['wmp_first'] = cong_df['subject_id'].isin(WM_FIRST)

    # Merge on subject_id + session_type
    merged = cong_df.merge(behav, on=['subject_id', 'session_type'], how='inner')

    # Reshape to long format matching the existing main_results.csv schema
    rows = []
    for _, row in merged.iterrows():
        base = {
            'subject_id':     row['subject_id'],
            'session_number': row.get('session_number', np.nan),
            'session_type':   row['session_type'],
            'n_voxels':       np.nan,
            'wmp_first':      row['wmp_first'],
        }
        rows.append({**base, 'metric': 'delta_EVR', 'score': row['delta_EVR']})
        rows.append({**base, 'metric': 'delta_BC',  'score': row['delta_BC']})

    main_df = pd.DataFrame(rows)
    main_df.to_csv(MAIN_RESULTS)
    print(f'Saved {MAIN_RESULTS}  shape={main_df.shape}')

else:
    print(f'Loading existing {MAIN_RESULTS}')


# ── Load and prepare ──────────────────────────────────────────────────────────

sns.set_context(context_params)

main_df = pd.read_csv(MAIN_RESULTS, index_col=0).drop_duplicates()

# Exclude subjects not included in neural analyses
exclude_str = [f'avatarRT_sub_{s:02d}' for s in exclude_from_neural_analyses]
main_df = main_df[~main_df['subject_id'].isin(exclude_str)]

# Wide format: one row per subject × session_type
wide = (main_df
        .pivot_table(index=['subject_id', 'session_type', 'wmp_first'],
                     columns='metric', values='score')
        .reset_index())
wide.columns.name = None
print(f'Wide-format data: {wide.shape[0]} rows, subjects = {wide.subject_id.nunique()}')


# ── Plot 1: delta_EVR per session type (make_barplot_points) ─────────────────

fig, ax = make_barplot_points(
    wide, 'delta_EVR', 'session_type',
    exclude_subs=exclude_from_neural_analyses,
    ylim=[-0.045, 0.08], outfn=None, title='',
    plus_bot=0.03, plus_top=0.055, n_iter=10000,
    sample_alternative='greater', pairwise_alternative='greater',
    ylabel='Δ Explained Variance Ratio', xlabel='',
)
plt.savefig(os.path.join(PLOTS_DIR, 'neural_EVR_barplot.pdf'),
            transparent=True, bbox_inches='tight', format='pdf')
plt.show()


# ── Mixed-effects model: delta_BC ~ delta_EVR * session_type ────────────────

# Random intercept per subject; wmp_first is a between-subjects factor (constant
# within each subject across sessions) so it is included as a fixed covariate —
formula = 'delta_BC ~ 0 + delta_EVR*C(session_type) + wmp_first'
me_model = smf.mixedlm(formula,
    data=wide, groups=wide['subject_id'],
).fit(reml=True)

print('\n=== Mixed effects: {formula} ===')
print(me_model.summary())

# Per-session-type simple slopes (OLS within each condition)
simple_slope_rows = []
for M in ORDER:
    sub = wide[wide['session_type'] == M].dropna(subset=['delta_EVR', 'delta_BC'])
    ols = smf.ols('delta_BC ~ delta_EVR', data=sub).fit()
    beta = ols.params['delta_EVR']
    pval = ols.pvalues['delta_EVR']
    ci   = ols.conf_int().loc['delta_EVR']
    r2   = ols.rsquared
    simple_slope_rows.append({
        'session_type': M,
        'beta_delta_EVR': beta,
        'std_err': ols.bse['delta_EVR'],
        'CI_lower': ci[0], 'CI_upper': ci[1],
        'p_value': pval,
        'R2': r2,
        'n': len(sub),
    })
    print(f'{M}: β={beta:.4f}, p={pval:.4f}, R²={r2:.3f}')

simple_slopes_df = pd.DataFrame(simple_slope_rows)

# Save mixed-model coefficients + simple slopes to CSV
fe_idx = me_model.fe_params.index
ci = me_model.conf_int().loc[fe_idx]
me_coefs_df = pd.DataFrame({
    'term':     fe_idx,
    'coef':     me_model.fe_params.values,
    'std_err':  me_model.bse_fe.values,
    'z':        me_model.tvalues.loc[fe_idx].values,
    'p_value':  me_model.pvalues.loc[fe_idx].values,
    'CI_lower': ci.iloc[:, 0].values,
    'CI_upper': ci.iloc[:, 1].values,
})
me_coefs_df.to_csv(os.path.join(RESULTS_PUBLIC, 'neural_EVR_mixedlm_coefs.csv'), index=False)
simple_slopes_df.to_csv(os.path.join(RESULTS_PUBLIC, 'neural_EVR_simple_slopes.csv'), index=False)
print(f'\nSaved mixed-model results to {RESULTS_PUBLIC}')
print(simple_slopes_df[['session_type', 'beta_delta_EVR', 'p_value', 'R2']].to_string(index=False))


# ── Plot 2: 3-panel scatter + regression with CI (shared axes) ───────────────

# Compute global axis limits across all conditions
plot_data = wide.dropna(subset=['delta_EVR', 'delta_BC'])
x_pad = (plot_data['delta_EVR'].max() - plot_data['delta_EVR'].min()) * 0.08
y_pad = (plot_data['delta_BC'].max()  - plot_data['delta_BC'].min())  * 0.08
xlim = (plot_data['delta_EVR'].min() - x_pad, plot_data['delta_EVR'].max() + x_pad)
ylim = (plot_data['delta_BC'].min()  - y_pad, plot_data['delta_BC'].max()  + y_pad)

fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), sharex=True, sharey=True)

for i, (M, ax) in enumerate(zip(ORDER, axes)):
    sub = wide[wide['session_type'] == M].dropna(subset=['delta_EVR', 'delta_BC'])
    row = simple_slopes_df[simple_slopes_df['session_type'] == M].iloc[0]

    # Regression line + 95% CI band (no scatter from regplot)
    sns.regplot(x='delta_EVR', y='delta_BC', data=sub, ax=ax,
                scatter=False, ci=95,
                color=colors_main[M],
                line_kws={'linewidth': 1.8, 'zorder': 5},
                )

    # Subject scatter points on top
    ax.scatter(sub['delta_EVR'], sub['delta_BC'],
               color=colors_sim[M], edgecolors='k', linewidths=0.6,
               s=40, zorder=10, alpha=0.9)

    # Reference lines
    ax.axhline(0, ls='--', c='k', lw=0.8)
    ax.axvline(0, ls='--', c='k', lw=0.8)

    # Annotation: beta + p-value
    symb = determine_symbol(row['p_value'])
    pstr = f"p={row['p_value']:.3f}" if row['p_value'] >= 0.001 else "p<0.001"
    ax.annotate(f"β={row['beta_delta_EVR']:.3f}\n{pstr}  {symb}",
                xy=(0.05, 0.95), xycoords='axes fraction',
                va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

    ax.set(xlim=xlim, ylim=ylim, title=M, xlabel='Δ EVR')
    if i == 0:
        ax.set_ylabel('Δ Brain Control', fontsize=10)
    else:
        ax.set_ylabel('')

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'neural_EVR_regression_panels.pdf'),
            transparent=True, bbox_inches='tight', format='pdf')
plt.show()

print(f'\nAll plots saved to {PLOTS_DIR}')
print('Neural EVR results complete.')
