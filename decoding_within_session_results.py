# decoding_within_session_results.py
# Plots MSE of decoders trained and tested within the same session type,
# as in revision.ipynb cell 53 (make_barplot_points on df_cong_avg).
#
# Source file: results/results_public/decoding_results_aug6_cross_session_run_cross_validation.csv
# Output plot: results/plots/decoders_within_session.pdf

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["pdf.use14corefonts"] = True
import matplotlib.pyplot as plt
import seaborn as sns
from plotting_functions import make_barplot_points
from config import *

RESULTS_PUBLIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'results_public')
PLOTS_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'plots')
DECODING_FN    = os.path.join(RESULTS_PUBLIC, 'decoding_results_aug6_cross_session_run_cross_validation.csv')

os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_context(context_params)


def main():
    if not os.path.exists(DECODING_FN):
        print(f'File not found: {DECODING_FN}')
        return

    df = pd.read_csv(DECODING_FN, index_col=0)
    df = df[~df['subject_id'].isin(['avatarRT_sub_09', 'avatarRT_sub_20'])]

    # Within-session rows: trained and tested on the same session type
    df_cong = df[df['congruent'] == True].copy()

    # Average across folds and runs → one MSE value per subject × session type
    df_cong_avg = (df_cong
                   .groupby(['subject_id', 'test_session_type'], as_index=False)['mse']
                   .mean()
                   .rename(columns={'test_session_type': 'session_type'}))

    print(f'Within-session data: {df_cong_avg.shape[0]} rows '
          f'({df_cong_avg["subject_id"].nunique()} subjects × 3 session types)')
    print(df_cong_avg.groupby('session_type')['mse'].describe().round(3))

    out_fn = os.path.join(PLOTS_DIR, 'decoders_within_session.pdf')
    make_barplot_points(
        df_cong_avg, 'mse', 'session_type',
        exclude_subs=[9, 20],
        ylim=[0, 1.5],
        outfn=out_fn,
        title='Decoders evaluated within session',
        plus_bot=0.2, plus_top=0.35,
        n_iter=10000,
        sample_alternative='greater',
        pairwise_alternative='two-sided',
        ylabel='MSE', xlabel='Session type',
    )
    print(f'Saved plot: {out_fn}')


if __name__ == '__main__':
    main()
