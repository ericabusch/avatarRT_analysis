## This script runs comparisons along the neurofeedback components
import numpy as np
import pandas as pd
import os, sys, glob, shutil
import analysis_helpers as helper
import argparse
from scipy.stats import zscore, wasserstein_distance
from sklearn.decomposition import PCA
import analysis_helpers as helper
from config import *

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
            XTarget = helper.load_component_data(subject_id ,  ses_id, run, component_number=None, by_trial=False, shift_by=2)
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


if __name__ == '__main__':

    ################ Parse CL args #######################
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", '--verbose',type=int, default=1)
    args = parser.parse_args()
    VERBOSE=args.verbose

    cumulative_info = helper.load_info_file()
    result_list_overall, subtraction_list_overall = [], []

    for subject_id  in SUB_IDS:
        subject_info = cumulative_info[cumulative_info['subject_ID'] == subject_id ]
        
        # loads the 0th index component (the IM)
        im_session = int(subject_info['im_session'].item())
        im_component, im_idx = 0, 0

        # loads the WMP; adjusts for 0 indexing (but that 0 is the IM, so 1=1)
        wmp_session = int(subject_info['wmp_session'].item())
        wmp_component=int(subject_info['wmp_component'].item())
        wmp_idx = wmp_component
        if wmp_component != 1: wmp_idx = wmp_component - 1
        else: wmp_idx = 1

        # loads the OMP; adjusts for 0 indexing (but that 0 is the IM, so 1=1)
        omp_session = int(subject_info['omp_session'].item())
        omp_component=int(subject_info['omp_component'].item())
        omp_idx = omp_component - 1 

        session_name_mapping = {'IM': [im_session, im_idx, im_component], 
        'WMP':[wmp_session, wmp_idx, wmp_component], 
        'OMP':[omp_session, omp_idx, omp_component]}

        if VERBOSE: print(f'{subject_id }  im,wmp,omp ses = {im_session},{wmp_session},{omp_session}')

        subjPath=f'{DATA_PATH}/{subject_id}' 
        outfn=f'{subjPath}/results/runwise_component_EVR_neural_analysis_final'
        results, subtraction_results = delta_evr_across_components(subject_id, session_name_mapping)

        results['subject_id'] = subject_id 
        results.to_csv(outfn+'.csv')
        result_list_overall.append(results)

        subtraction_results['subject_id'] = subject_id 
        subtraction_list_overall.append(subtraction_results)

    res_overall = pd.concat(result_list_overall).reset_index(drop=True)
    sub_overall = pd.concat(subtraction_list_overall).reset_index(drop=True)
    res_overall.to_csv(f'{FINAL_RESULTS_PATH}/runwise_component_EVR_neural_analysis_by_trial.csv')
    sub_overall.to_csv(f'{FINAL_RESULTS_PATH}/runwise_component_EVR_neural_analysis_run_change.csv')
    

