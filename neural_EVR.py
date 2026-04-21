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
from scipy.stats import chi2
import analysis_helpers as helper
from plotting_functions import make_barplot_points_precomputed, determine_symbol
from config import *

RESULTS_PUBLIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'results_public')
PLOTS_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'plots')
MAIN_RESULTS   = os.path.join(FINAL_RESULTS_PATH, 'main_results.csv')

os.makedirs(PLOTS_DIR, exist_ok=True)
def delta_evr_across_components(subject_id, session_name_mapping):

    results = pd.DataFrame(columns=['session_type', 'session_id', 'nfb_component_idx','congruent', 
        'comparison_component_idx','score','run', 'comparison_component_name'])
    
    subtraction_results = pd.DataFrame(columns=['session_type','session_number','nfb_component_idx', 'congruent', 
        'comparison_component_idx', 'delta_run_perturb', 'comparison_component_name'])
    
    for ses_name, ses_info in session_name_mapping.items():
        ses_id = f'ses_{ses_info[0]:02d}'
        run_numbers = [1,2,3,4]
        for run in run_numbers:
            # this will return projected onto the potential NFB comps
            XTarget = helper.load_component_data(subject_id , ses_id, run, component_number=None, by_trial=False, shift_by=2)
            evr_mat = helper.run_EVR(XTarget)
            for comparison_comp, comparison_info in session_name_mapping.items():
                idx = comparison_info[1]
                ev = evr_mat[idx]
                results.loc[len(results)] = {'session_type':ses_name, 
                'session_id':ses_id, 
                'nfb_component_idx': ses_info[1], 
                'congruent':comparison_comp == ses_name,
                'comparison_component_idx':idx,
                'comparison_component_name':comparison_comp,
                'run':run,
                'score':ev}
                
        # compare across runs, within this session, shift, etc
        first_run, last_run = SESSION_TYPES_RUNS[ses_name][0], SESSION_TYPES_RUNS[ses_name][-1]
        for comparison_comp, comparison_info in session_name_mapping.items():
            temp = results[results['session_id']==ses_id].reset_index(drop=True) 
            idx = comparison_info[1]
            C = ['run','comparison_component_name']
            first_run_data = helper.extract_data_from_df(temp, C, [first_run, comparison_comp],"score")
            last_run_data = helper.extract_data_from_df(temp, C, [last_run, comparison_comp],"score")
            main_diff = last_run_data[-1] - first_run_data[0]              
            session_number = ses_id
            if subject_id == 'avatarRT_sub_06': 
                session_number=f'ses_{ses_info[0]-1:02d}'
            subtraction_results.loc[len(subtraction_results)] = {'session_type':ses_name,
                        'nfb_component_idx':ses_info[1], 
                        'congruent':comparison_comp == ses_name, 
                        'comparison_component_idx':comparison_info[1], 
                        'delta_run_perturb':main_diff,
                        'session_number':session_number,
                        'comparison_component_name':comparison_comp} 
    return results, subtraction_results       

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
        os.path.join(FINAL_RESULTS_PATH, 'behavioral_change_runwise_with_simulations.csv'),
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


# Pairwise between-session-type comparisons on delta_EVR (subjects matched via pivot)
d_wide_evr = wide.pivot(index='subject_id', columns='session_type', values='delta_EVR').dropna()
v_im = d_wide_evr['IM'].values
v_wm = d_wide_evr['WMP'].values
v_om = d_wide_evr['OMP'].values
print("\n--- delta_EVR comparisons wiht zero---")
zeros = np.zeros(len(v_im))  # for paired permutation tests against 0
_, p_im_0, _  = helper.permutation_test(np.array([v_im, zeros]), 10000, alternative='greater')
_, p_wmp_0, _ = helper.permutation_test(np.array([v_wm, zeros]), 10000, alternative='greater')
_, p_omp_0, _ = helper.permutation_test(np.array([v_om, zeros]), 10000, alternative='two-sided')
m_im_0,  lower_im_0,  upper_im_0,  _ = helper.bootstrap_ci(v_im, n_boot=10000, verbose=0)
m_wmp_0, lower_wmp_0, upper_wmp_0, _ = helper.bootstrap_ci(v_wm, n_boot=10000, verbose=0)
m_omp_0, lower_omp_0, upper_omp_0, _ = helper.bootstrap_ci(v_om, n_boot=10000, verbose=0)
d_im_0  = helper.cohens_d_paired(v_im, verbose=0)
d_wmp_0 = helper.cohens_d_paired(v_wm, verbose=0)
d_omp_0 = helper.cohens_d_paired(v_om, verbose=0)
print(f'IM   vs 0:   mean={m_im_0*100:.2f}  p={p_im_0:.2f}    95%CI=[{lower_im_0*100:.2f},{upper_im_0*100:.2f}]    d={d_im_0:.2f}')
print(f'WMP  vs 0:   mean={m_wmp_0*100:.2f} p={p_wmp_0:.2f}   95%CI=[{lower_wmp_0*100:.2f},{upper_wmp_0*100:.2f}]  d={d_wmp_0:.2f}')
print(f'OMP  vs 0:   mean={m_omp_0*100:.2f} p={p_omp_0:.2f}   95%CI=[{lower_omp_0*100:.2f},{upper_omp_0*100:.2f}]  d={d_omp_0:.2f}')

_, p_im_wm, _ = helper.permutation_test(np.array([v_im, v_wm]), 10000, alternative='two-sided')
_, p_im_om, _ = helper.permutation_test(np.array([v_im, v_om]), 10000, alternative='greater')
_, p_wm_om, _ = helper.permutation_test(np.array([v_wm, v_om]), 10000, alternative='greater')
m_im_wm, lo_im_wm, hi_im_wm, _ = helper.bootstrap_ci(v_im - v_wm, n_boot=10000, verbose=0)
m_im_om, lo_im_om, hi_im_om, _ = helper.bootstrap_ci(v_im - v_om, n_boot=10000, verbose=0)
m_wm_om, lo_wm_om, hi_wm_om, _ = helper.bootstrap_ci(v_wm - v_om, n_boot=10000, verbose=0)
d_im_wm = helper.cohens_d_paired(v_im - v_wm, verbose=0)
d_im_om = helper.cohens_d_paired(v_im - v_om, verbose=0)
d_wm_om = helper.cohens_d_paired(v_wm - v_om, verbose=0)
print("\n--- delta_EVR pairwise comparisons ---")
print(f"  IM  vs WMP:  mean={m_im_wm*100:.2f}  p={p_im_wm:.2f}  95%CI=[{lo_im_wm*100:.2f},{hi_im_wm*100:.2f}]  d={d_im_wm:.2f}")
print(f"  IM  vs OMP:  mean={m_im_om*100:.2f}  p={p_im_om:.2f}  95%CI=[{lo_im_om*100:.2f},{hi_im_om*100:.2f}]  d={d_im_om:.2f}")
print(f"  WMP vs OMP:  mean={m_wm_om*100:.2f}  p={p_wm_om:.2f}  95%CI=[{lo_wm_om*100:.2f},{hi_wm_om*100:.2f}]  d={d_wm_om:.2f}")

evr_stats = []
for label, vals, p, cohd, ci_lo, ci_hi, alt in [
    ('IM',  v_im, p_im_0,  d_im_0,  lower_im_0,  upper_im_0,  'greater'),
    ('WMP', v_wm, p_wmp_0, d_wmp_0, lower_wmp_0, upper_wmp_0, 'greater'),
    ('OMP', v_om, p_omp_0, d_omp_0, lower_omp_0, upper_omp_0, 'two-sided'),
]:
    evr_stats.append({
        'comparison': f'delta_EVR: {label} vs 0',
        'test': f'permutation_test (n_iter=10000, alternative={alt})',
        'group1': label, 'group2': '0 (null)',
        'n1': len(vals), 'n2': np.nan,
        'mean1': np.mean(vals), 'mean2': 0,
        'p_value': p, 'ci_lower': ci_lo, 'ci_upper': ci_hi,
        'cohens_d': cohd,
    })
for label, g1, g2, g1v, g2v, diff_vals, p, cohd, ci_lo, ci_hi, alt in [
    ('IM vs WMP', 'IM',  'WMP', v_im, v_wm, v_im - v_wm, p_im_wm, d_im_wm, lo_im_wm, hi_im_wm, 'two-sided'),
    ('IM vs OMP', 'IM',  'OMP', v_im, v_om, v_im - v_om, p_im_om, d_im_om, lo_im_om, hi_im_om, 'greater'),
    ('WMP vs OMP','WMP', 'OMP', v_wm, v_om, v_wm - v_om, p_wm_om, d_wm_om, lo_wm_om, hi_wm_om, 'greater'),
]:
    evr_stats.append({
        'comparison': f'delta_EVR: {label}',
        'test': f'permutation_test (n_iter=10000, alternative={alt})',
        'group1': g1, 'group2': g2,
        'n1': len(g1v), 'n2': len(g2v),
        'mean1': np.mean(g1v), 'mean2': np.mean(g2v),
        'p_value': p, 'ci_lower': ci_lo, 'ci_upper': ci_hi,
        'cohens_d': cohd,
    })
evr_stats_df = pd.DataFrame(evr_stats)
evr_stats_df['significant_0.05'] = evr_stats_df['p_value'] < 0.05
evr_stats_fn = os.path.join(FINAL_RESULTS_PATH, 'neural_EVR_permutation_stats.csv')
evr_stats_df.to_csv(evr_stats_fn, index=False)
print(f'Saved EVR permutation stats to {evr_stats_fn}')

# ── Plot 1: delta_EVR per session type ───────────────────────────────────────
fig, ax = make_barplot_points_precomputed(
    wide, 'delta_EVR', 'session_type',
    pvals_vs_0=[p_im_0, p_wmp_0, p_omp_0],
    pvals_pairwise=[p_im_wm, p_im_om, p_wm_om],
    exclude_subs=exclude_from_neural_analyses,
    ylim=[-0.045, 0.08],
    plus_bot=0.03, plus_top=0.055,
    ylabel='Δ Explained Variance Ratio', xlabel='',
)
plt.savefig(os.path.join(PLOTS_DIR, 'neural_EVR_barplot.pdf'),
            transparent=True, bbox_inches='tight', format='pdf')


# ── Mixed-effects model: delta_BC ~ delta_EVR * session_type ────────────────
# Compare the two models, choose the simpler one if the random intercepts is not significant, and report the results.
# Fit both models
lme_model = smf.mixedlm('delta_BC ~ delta_EVR * C(session_type) + wmp_first', 
                         data=wide, groups=wide['subject_id']).fit(reml=False)
ols_model = smf.ols('delta_BC ~ delta_EVR * C(session_type) + wmp_first', 
                     data=wide).fit()

# Likelihood ratio test
LR_stat = 2 * (lme_model.llf - ols_model.llf)
df_diff = 1  # Difference in number of parameters (random intercept variance)
p_value = 0.5  * chi2.sf(LR_stat, df=df_diff)  # df=1 for one random effect parameter, variance is on the boundary of parameter space (non-negative)
print(f"Comparing LME and OLS models; LR statistic: {LR_stat:.3f}, p = {p_value:.3f}")

# Random intercept per subject; wmp_first is a between-subjects factor (constant
# within each subject across sessions) so it is included as a fixed covariate —
# formula = 'delta_BC ~ 0 + delta_EVR*C(session_type) + wmp_first'
# me_model = smf.mixedlm(formula,
#     data=wide, groups=wide['subject_id'],
# ).fit(reml=True) 
formula = '0 + delta_BC ~ delta_EVR * C(session_type) + wmp_first'
ols_model = smf.ols(formula, 
                     data=wide).fit(cov_type='cluster', 
                                  cov_kwds={'groups': wide['subject_id']})

print(f'\n=== Model: {formula} ===')
print(ols_model.summary())
print("\n--- ANOVA on ols model () ---")
anova_results = anova_lm(ols_model, typ=3)
print(anova_results[['sum_sq', 'df', 'F', 'PR(>F)']])
ss_residual = anova_results.loc['Residual', 'sum_sq']
anova_results['partial_eta_sq'] = anova_results['sum_sq'] / (anova_results['sum_sq'] + ss_residual)
anova_results['R2_full_model'] = ols_model.rsquared
anova_fn = os.path.join(FINAL_RESULTS_PATH, 'neural_EVR_anova_results.csv')
anova_results.to_csv(anova_fn)
print(f'Saved ANOVA results (with partial η²) to {anova_fn}')

# Per-session-type simple slopes (OLS within each condition)
simple_slope_rows = []
for M in ORDER:
    sub = wide[wide['session_type'] == M].dropna(subset=['delta_EVR', 'delta_BC'])
    ols = smf.ols('delta_BC ~ delta_EVR', data=sub).fit()
    print(f'\n--- Simple slope for {M} ---')
    print(ols.summary())
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
        'partial_eta_sq': np.nan,
        'n': len(sub),
    })

    # print(f'{M}: β={beta:.2f}, p={pval:.2f}, R²={r2:.3f}')

simple_slopes_df = pd.DataFrame(simple_slope_rows)

# Save model coefficients + simple slopes to CSV
fe_idx = ols_model.params.index
ci = ols_model.conf_int().loc[fe_idx]
me_coefs_df = pd.DataFrame({
    'term':     fe_idx,
    'coef':     ols_model.params.values,
    'std_err':  ols_model.bse.values,
    'z':        ols_model.tvalues.loc[fe_idx].values,
    'p_value':  ols_model.pvalues.loc[fe_idx].values,
    'CI_lower': ci.iloc[:, 0].values,
    'CI_upper': ci.iloc[:, 1].values,
})
me_coefs_df['R2_full_model'] = ols_model.rsquared
me_coefs_df.to_csv(os.path.join(FINAL_RESULTS_PATH, 'neural_EVR_lm_coefs.csv'), index=False)
simple_slopes_df.to_csv(os.path.join(FINAL_RESULTS_PATH, 'neural_EVR_simple_slopes.csv'), index=False)
print(f'\nSaved model results to {FINAL_RESULTS_PATH}')


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
# plt.show()

print(f'\nAll plots saved to {PLOTS_DIR}')
print('Neural EVR results complete.')
