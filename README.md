# avatarRT_analysis
*updated March 2026, ELB, with supplemental analyses after revision*

This repository contains the analysis scripts for a real-time fMRI experiment testing neural manifold constraints on learning.

---

## Environment

All scripts require the `rtcloud_av1` conda environment:

```bash
conda activate rtcloud_av1
# Interpreter: /Users/elb/miniconda3/envs/rtcloud_av1/bin/python
```

Run scripts from the `avatarRT_analysis/` directory:

```bash
cd /path/to/avatarRT_analysis
python behavioral_results.py
```

### Special package requirements

| Package | Purpose | Notes |
|---|---|---|
| `nibabel` | Load NIfTI mask files | Standard conda install |
| `ot` (POT) | Gromov-Wasserstein distances | `pip install POT` |
| `TPHATE` | T-PHATE manifold embedding | Local package in `TPHATE/` subdirectory |
| `MRAE` | Neural manifold model (bottleneck encoder) | Local package in `MRAE/` subdirectory — install with `pip install -e MRAE/` |
| `statsmodels` | OLS / ANOVA for order effects | Standard conda install |
| `himalaya` | Ridge regression (location decoding) | `pip install himalaya` |

---

## Configuring paths (`config.py`)

Edit `config.py` before running any scripts. The key path to set is:

```python
DATA_PATH = os.path.expanduser('~/Desktop/BCI/avatarRT_dryad/avatarRT_subject_data')
```

This should point to the root of the subject data directory. Each subject's folder is expected to follow this structure:

```
<DATA_PATH>/
  avatarRT_sub_05/
    model/
      
    reference/
      mask.nii.gz             
    ses_0X/
      run_Y/            # run-level data (behavioral, functional)
        ...                   
```

The `SESSION_TRACKER` CSV (at `<DATA_PATH>/session_tracker.csv`) maps each subject to their IM/WMP/OMP session numbers and component indices.

Other paths (`SCRATCH_PATH`, `INTERMEDIATE_RESULTS_PATH`, `FINAL_RESULTS_PATH`) are set relative to the script directory and do not normally need to be changed.

---

## Results directory structure

Scripts write output to `results/` (created automatically):

```
results/
  results_public/       # CSVs shared as supplementary data
  final_results/        # final per-subject summary CSVs (some pre-computed)
  intermediate_results/ # intermediate cached computations
  plots/                # all PDF figures
  scratch/              # temporary cached embeddings
```

---

## Statistical approach

All within-subjects comparisons use **nonparametric paired permutation tests** (10,000 iterations) via `helper.permutation_test`. Confidence intervals for vs-0 tests are computed with `helper.bootstrap_ci` (10,000 bootstrap samples). Between-subjects group comparisons (for real and simulated groups) use `scipy.stats.ttest_ind(permutations=10000)`.

---

## Scripts

### Core analysis scripts

#### `behavioral_results.py`
BCI learning effects indexed by behavior ($\Delta$ Brain Control).

**Outputs — plots:**
- `results/plots/behavioral_learning_sim.pdf` — observed vs. simulated delta-BC per session type
- `results/plots/behavioral_learning_trialseries.pdf` — trial-by-trial learning curves
- `results/plots/behavioral_learning_runwise.pdf` — run-wise learning curves

**Outputs — CSVs (`results/results_public/`):**
- `behavioral_stats.csv` — permutation test p-values and bootstrap CIs per condition

**Requires:** `results/final_results/behavioral_change_session.csv`, `results/final_results/behavioral_change_trialseries_with_simulations.csv`, `results/final_results/behavioral_change_runwise_with_simulations.csv`

---

#### `neural_EVR_results.py`
Neural alignment with manifold components (Explained Variance Ratio), across sessions and runs.

**Outputs — plots:**
- `results/plots/neural_EVR_*.pdf` — EVR barplots per session type and run

**Requires:** pre-computed EVR CSVs in `results/final_results/`

---

#### `joystick_results.py`
Spatial decoding of avatar location during the joystick task, and RSA analysis.
---

#### `decoding_within_session_results.py`
Within-session decoding performance (run-wise cross-validation).
---

#### `realignment_consolidation_results.py`
Evidence for neural reconsolidation after learning.

Two analyses:
1. **EVR resampling** — z-scores delta-EVR of the NFB feedback component vs. a null from all other components (10,000 draws).
2. **Cross-session decoding** — IM-trained decoders evaluated across sessions; delta_mse (run 4 − run 1).

**Outputs — plots:**
- `results/plots/consolidation_zscored_evr.pdf`
- `results/plots/consolidation_null_evr.pdf`
- `results/plots/consolidation_delta_decoding.pdf`

**Outputs — CSVs:**
- `results/results_public/random_resampling_components.csv`
- `results/results_public/delta_decoding.csv`

**Requires (for computing, not loading):** raw run data via `analysis_helpers`; optionally `results/final_results/runwise_component_EVR_neural_analysis_run_change_control.csv`

---

#### `order_effects_results.py`
Effect of counterbalancing order (WMP-first vs. OMP-first) on delta_BC and delta_EVR.

**Outputs — plots:**
- `results/plots/order_effects_barplot.pdf`

**Outputs — CSVs:**
- `results/results_public/order_effects_lm_results.csv`

**Requires:** `results/final_results/behavioral_change_session.csv`, `results/results_public/main_results.csv`

---

### Supplemental / revision analyses

#### `eigenspectrum_results.py`
Control-space eigenspectrum: percent variance explained per manifold component per subject.

**Outputs — plots:**
- `results/plots/eigenspectrum_grid.pdf` — per-subject component eigenspectrum
- `results/plots/eigenspectrum_by_condition.pdf` — PEV by NFB condition
- `results/plots/eigenspectrum_correlations.pdf` — PEV difference vs. learning outcomes

**Outputs — CSVs:**
- `results/results_public/manifold_eigenspectrum.csv`

**Requires:** `<DATA_PATH>/<subject>/model/bottleneck.npy` for each subject; optionally `results/results_public/main_results.csv` for correlation plots.

---

#### `mask_size_results.py`
Per-subject neurofeedback mask size (voxel count) and its correlation with BCI/neural learning.

**Outputs — plots:**
- `results/plots/mask_size.pdf`
- `results/plots/mask_size_learning.pdf`

**Outputs — CSVs:**
- `results/results_public/mask_size_per_subject.csv`
- `results/results_public/mask_size_correlations.csv`

**Requires:** `<DATA_PATH>/<subject>/reference/mask.nii.gz` for each subject; `results/results_public/main_results.csv`

---

#### `intrinsic_manifold_stability.py`
Stability of the intrinsic neural manifold across days using Gromov-Wasserstein distances.

**Outputs — plots:**
- `results/plots/manifold_stability.pdf`

**Outputs — CSVs:**
- `results/results_public/revision_gromov_wasserstein.csv`
- `results/results_public/gromov_wasserstein_analysis_results.csv`

**Requires:** `ot` (POT) and `TPHATE` packages; raw voxel data via `analysis_helpers`. T-PHATE embeddings are cached in `results/scratch/joystick_analyses/` after first computation.

---

#### `neural_manifold_variance_analysis.py`
Additional variance analysis on the neural manifold.

---

#### `searchlight_location_prediction_himalaya.py`
Searchlight spatial decoding of avatar location (requires `himalaya` package).

**Requires:** `run_randomise.sh` for group-level FSL randomise statistics over searchlight maps.

---

### Support modules

#### `analysis_helpers.py`
Shared statistical and data-loading utilities. Key functions:
- `permutation_test(data, n_iterations, alternative)` — sign-flip paired permutation test; `data` shape `(2, n_samples)`; returns `(observed, pvalue, null_distribution)`
- `bootstrap_ci(data, n_boot, ci, seed)` — bootstrap CI of the mean; returns `(mean, lower, upper)`
- `load_component_data(...)`, `run_EVR(...)`, `load_all_joystick_data(...)` — data loading helpers

#### `plotting_functions.py`
Shared plotting utilities. Key functions:
- `make_barplot_points(df, y, x, ...)` — barplot with individual points, permutation tests, and significance annotations
- `make_barplot_errorbar(df, y, x, ...)` — barplot with bootstrap error bars
- `determine_symbol(p)` — converts p-value to significance symbol

#### `config.py`
Global configuration: paths, subject lists, session parameters, colors, plot style.

---

## Excluded subjects

- `avatarRT_sub_12`: dropped out of the study
- `avatarRT_sub_09`: scanner issue (excluded from most analyses)
- `avatarRT_sub_20`: fell asleep during scan (excluded from most analyses)

Subjects 09 and 20 are excluded via the `SUBJECTS` list at the top of each analysis script. Subject 12 is absent from `SUB_NUMBERS` in `config.py`.
