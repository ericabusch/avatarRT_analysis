# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Use the `rtcloud_av1` conda environment for all scripts:

```bash
conda activate rtcloud_av1
# Interpreter: /Users/elb/miniconda3/envs/rtcloud_av1/bin/python
```

Build from scratch if needed (edit the last line of `environment.yml` to match your conda path first):
```bash
conda env create -f environment.yml
pip install tphate
pip install -e MRAE/
pip install POT
pip install himalaya
```

All analysis scripts must be run from the `avatarRT_analysis/` directory:
```bash
python <script_name>.py
```

## Configuration

**Before running anything**, set `DATA_PATH` in `config.py` to point to the root of the subject data directory (e.g. `~/Desktop/BCI/avatarRT_dryad/avatarRT_subject_data`). All other paths (`SCRATCH_PATH`, `INTERMEDIATE_RESULTS_PATH`, `FINAL_RESULTS_PATH`) are derived relative to the script directory and do not need to be changed.

The `SESSION_TRACKER` CSV lives at `<DATA_PATH>/session_tracker.csv` and maps each subject to their IM/WMP/OMP session numbers and NFB component indices.

## Architecture

### Data flow

Scripts generally follow a **load-or-compute** pattern: check whether a results CSV already exists, load it if so, otherwise run the full computation and save. Intermediate computations (e.g., T-PHATE embeddings) are cached as `.npy` files in `results/scratch/`.

### Key modules

- **`config.py`** — All global constants: paths, subject lists (`SUB_IDS`, `WM_FIRST`), excluded subjects, session parameters (`SHIFTBY=2`, `CALIB_TR=10`), colors, and plot style (`context_params`). Every analysis script does `from config import *`.

- **`analysis_helpers.py`** — The single source of truth for data loading and statistics. All analysis scripts import it as `import analysis_helpers as helper`. Key functions:
  - `permutation_test(data, n_iterations, alternative)` — sign-flip paired randomization test; `data` shape `(2, n_subjects)`
  - `bootstrap_ci(data, n_boot, ci, seed)` — bootstrap CI of the mean
  - `cohens_d_paired(data)` / `cohens_d_independent(g1, g2)` — effect sizes
  - `load_all_joystick_data(subject_id, run, mask, shift_by)` — loads/normalizes joystick-session voxel data
  - `get_realtime_data_preprocesssed(subject_id, session_id, run)` — offline-preprocessed RT data
  - `embed_tphate(X, t=5, n_components=2)` — wraps `tphate.TPHATE(...).fit_transform(X)`; pass `n_components=20` for manifold analyses
  - `load_component_data(...)`, `calculate_nfb_component_loadings(...)`, `load_model_from_dir(...)` — NFB model/component loading
  - `get_trial_data(X, subject_id, session_id, run, shift_by)` — splits timeseries into per-trial dicts with BOLD lag applied
  - `shift_timing(nTRs, labels, shift_size, start_label)` — applies hemodynamic shift to label vectors

- **`plotting_functions.py`** — Shared figure utilities, always used via `from plotting_functions import ...`. Internally calls `helper.permutation_test` and `helper.bootstrap_ci`. Key functions:
  - `make_barplot_points(df, yname, xname, ...)` — barplot + individual subject points + significance
  - `make_barplot_errorbar(df, yname, xname, ...)` — barplot + bootstrap error bars
  - `determine_symbol(p)` — converts p-value to `~`, `*`, `**`, `***`, or `n.s.`

- **`MRAE/`** — Local Python package for the Manifold Regularized Autoencoder. Install with `pip install -e MRAE/`. Used via `from MRAE.mrae import ManifoldRegularizedAutoencoder` and `from MRAE import dataHandler`.

### Subject exclusions

- `avatarRT_sub_12`: dropped out — absent from `SUB_NUMBERS` in `config.py`
- `avatarRT_sub_09`, `avatarRT_sub_20`: scanner/behavioral issues — excluded at the top of each analysis script via `SUBJECTS = [s for s in SUB_IDS if s not in ['avatarRT_sub_09', 'avatarRT_sub_20']]`

### Session types

Each subject has three RT sessions:
- **IM** (intrinsic mapping) — always run 1 within a session; baseline/calibration
- **WMP** (within-manifold perturbation) — feedback along a manifold-aligned component
- **OMP** (off-manifold perturbation) — feedback along an off-manifold component

Session assignments per subject are stored in `session_tracker.csv` and accessed via `helper.load_info_file()` or `helper.get_perturbation_info(subject_id, session_id, run)`.

### Statistical conventions

- Within-subjects comparisons: `helper.permutation_test` (10,000 iterations, sign-flip)
- Confidence intervals: `helper.bootstrap_ci` (10,000 samples)
- Between-subjects: `scipy.stats.ttest_ind(permutations=10000)`
- BOLD lag: always shift labels by `SHIFTBY=2` TRs (4 s)
- Calibration TRs: always exclude first `CALIB_TR=10` TRs from each run

## Results directory

```
results/
  results_public/   # source-data CSVs shared with publication
  final_results/    # per-subject summary CSVs (inputs for results scripts)
  intermediate_results/
  plots/            # PDF figures
  scratch/          # cached embeddings (.npy), created automatically
```
