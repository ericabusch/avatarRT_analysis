# Clean up behavioral results - final , both real and simulated subjects
import numpy as np
import pandas as pd
import os, sys, glob 
import matplotlib.pyplot as plt
import seaborn as sns
import analysis_helpers as helper
import inspect
from config import *

def normalized_delta_bc_trialseries(subject_id, subject_row):
    fn = f'{DATA_PATH}/{subject_id}/results/success_rates.csv'
    if not does_file_exist(fn): return
    df = pd.read_csv(fn, index_col=0).reset_index(drop=True) 

    if np.sum(np.isnan(df['game_control'].values)) > 0:
        print(f'{fn} contains nans')

    outfn = f'{INTERMEDIATE_RESULTS_PATH}/{subject_id}_trialwise_behavioral_results.csv'    
    # load in the trial error
    omp_ses = subject_row['omp_session'].item()
    wmp_ses = subject_row['wmp_session'].item()

    session_perturb_mapping = {'IM':2, "WMP":wmp_ses, "OMP":omp_ses}

    perturbation_types = np.zeros(len(df))
    iloc_wmp = df[(df['run_number']>=2) & (df['session_number']==wmp_ses)].index
    iloc_omp = df[(df['run_number']>=2) & (df['session_number']==omp_ses)].index
    if np.max(iloc_wmp) > len(perturbation_types): iloc_wmp = iloc_wmp[iloc_wmp<len(perturbation_types)]
    if np.max(iloc_omp) > len(perturbation_types): iloc_omp = iloc_omp[iloc_omp<len(perturbation_types)]
    
    perturbation_types[iloc_omp] = 2
    perturbation_types[iloc_wmp] = 1

    # turn into strings
    string_mapping = {0:'IM',1:'WMP',2:'OMP'}
    perturbation_strings = ['' for i in range(len(perturbation_types))]
    for i in range(len(perturbation_types)):
        perturbation_strings[i] = string_mapping[perturbation_types[i]]
    df['perturbation']=perturbation_strings

    # get error after each trial
    error_rates = np.zeros(len(perturbation_types))
    for i in range(len(df)):
       row = df.iloc[i]
       session_num = row.session_number
       #if subject_id == 'avatarRT_sub_06': session_num = row.session_number+1
       try:
            err = helper.get_trial_error(subject_id, f'ses_{session_num:02d}', row.run_number, row.round_number)
       except:
            print(subject_id, session_num, row.run_number, row.round_number, ' failed in getting trial error')
            err=np.nan
       error_rates[i] = err

    df['error']=error_rates
    df['brain_control']=1-df['game_control'].values
    normalized_perturb_start = np.zeros(len(df))

    # normalize BC 
    for p,s in session_perturb_mapping.items():

       # figure out the indices where we'll normalize
        idx = df[df['perturbation']==p].index
        values = df.iloc[idx]['brain_control'].values
        starting_run = SESSION_TYPES_RUNS[p][0]
        # first trial of first run
        normalize_to = helper.extract_data_from_df(df, ['session_number','run_number','round_number'], [s, starting_run, 0], 'brain_control')
        normalized_perturb_start[idx]=values-normalize_to
    df['brain_control_normalized_perturb_start']=normalized_perturb_start
    df['subject_id']=subject_id
    df.to_csv(outfn) 
    return df

def delta_BC_session(subject_id, subject_row, trialseries_df): 
    df1 = pd.DataFrame(columns=['delta_error','delta_BC','session_type','session_number'])
    session_perturb_mapping = {'IM':2, "WMP":subject_row['wmp_session'].item(), "OMP":subject_row['omp_session'].item()}
    for p, s in session_perturb_mapping.items():
        first_run = SESSION_TYPES_RUNS[p][0]
        final_run = SESSION_TYPES_RUNS[p][-1]

        init_bc=helper.extract_data_from_df(trialseries_df, ['session_number','run_number','round_number'], [s, first_run, 0], 'brain_control')[0]
        init_error=np.nanmean(helper.extract_data_from_df(trialseries_df, ['session_number','run_number'], [s, first_run], 'error'))

        # final trial number
        final_run_trials = helper.extract_data_from_df(trialseries_df, ['session_number','run_number'], [s, final_run], 'round_number')
        fin_trial = final_run_trials.max()

        fin_bc = helper.extract_data_from_df(trialseries_df, ['session_number','run_number','round_number'], [s, final_run, fin_trial], 'brain_control')[0]
        fin_error = np.nanmean(helper.extract_data_from_df(trialseries_df, ['session_number','run_number'], [s, final_run], 'error'))
        df1.loc[len(df1)] = {'delta_error': fin_error-init_error, 'delta_BC':  fin_bc - init_bc, 'session_type':p, 'session_number':s}
    df1['subject_id'] = [subject_id for i in range(len(df1))]
    return df1

if __name__ == '__main__':
    cumulative_info = helper.load_info_file()
    timeseries_dfs, overall_dfs = [], []
    for subject in SUB_IDS:
        row = cumulative_info[cumulative_info['subject_ID']==subject]
        d = normalized_delta_bc_trialseries(subject, row)
        timeseries_dfs.append(d)

        # now change to be difference between session start/end
        b=delta_BC_session(subject, row, d)
        overall_dfs.append(b)

    df = pd.concat(timeseries_dfs).reset_index(drop=True)
    df.to_csv(f'{FINAL_RESULTS_PATH}/behavioral_change_trialseries.csv')
    print(f'trialseries df of shape {df.shape}')


    df = pd.concat(overall_dfs).reset_index(drop=True)
    print(f'overall df of shape {df.shape}')
    df.to_csv(f'{FINAL_RESULTS_PATH}/behavioral_change_session.csv')





