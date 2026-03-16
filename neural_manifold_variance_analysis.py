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

def total_variance_change_prepost_learning(subject_id, session_name_mapping):

    results = pd.DataFrame(columns=['session_type', 'session_id', 'run', 'total_variance', 'average_variance', 'data_type'])
    
    subtraction_results = pd.DataFrame(columns=['session_type', 'session_id', 'delta_total_variance', 'delta_average_variance', 'data_type'])
    
    run_numbers = [1,2,3,4]

    # the nfb component idx is the component index being trained in a given run
    # the comparison component index is the component along which we're measuring change for this row
    # overall means run 1 - run 4, perturb is since the perturbation was introduced
    # comparison component name can be "IM',"WMP','OMP','untrained_{i}'    
    
    
    for ses_name, ses_info in session_name_mapping.items():
        ses_id = ses_info['session_number']
        ses_id=f'ses_0{ses_id}'
        
        first_run, last_run = SESSION_TYPES_RUNS[ses_name][0], SESSION_TYPES_RUNS[ses_name][-1]
        
        for run in run_numbers:
            # this will return projected onto the potential NFB comps
            comp_data = helper.load_component_data(subject_id , ses_id, run, component_number=None, by_trial=False, shift_by=2)
            voxel_data = helper.get_realtime_outdata(subject_id , ses_id, run, data_type="masked_data")
            mani_data = helper.load_manifold_projected_data(subject_id , ses_id, run)
            results.loc[len(results)] = {'session_type':ses_name, 
                                         'session_id':ses_id, 
                                         'run':run, 
                                         'total_variance': np.var(mani_data), 
                                         'average_variance':np.mean(np.var(mani_data, axis=0)), 
                                         'data_type':'manifold_projection'}
            results.loc[len(results)] = {'session_type':ses_name, 
                                         'session_id':ses_id, 
                                         'run':run, 
                                         'total_variance': np.var(voxel_data), 
                                         'average_variance':np.mean(np.var(voxel_data, axis=0)), 
                                         'data_type':'voxel_data'}
            results.loc[len(results)] = {'session_type':ses_name, 
                                         'session_id':ses_id, 
                                         'run':run, 
                                         'total_variance': np.var(comp_data), 
                                         'average_variance':np.mean(np.var(comp_data, axis=0)), 
                                         'data_type':'component_data'}
        
        for dt in ['manifold_projection','voxel_data','component_data']:
            temp = results[(results['session_type']==ses_name)&(results['data_type']==dt)]
            first_run_dat=temp[temp['run']==first_run]
            last_run_dat=temp[temp['run']==last_run]
            subtraction_results.loc[len(subtraction_results)] = {'session_type':ses_name, 
                                                                 'session_id':ses_id, 
                                                                 'delta_total_variance': last_run_dat['total_variance'].item() - first_run_dat['total_variance'].item(), 
                                                                 'delta_average_variance': last_run_dat['average_variance'].item() - first_run_dat['average_variance'].item(), 'data_type':dt}
        
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

        session_name_mapping = {
            'IM': {'session_number':im_session,
                   'component_index': im_idx, 
                                       'component_human_written': im_component, 
                                       'untrained_components':np.setdiff1d(np.arange(20), [im_idx, wmp_idx, omp_idx])}, 
        'WMP':{'session_number': wmp_session, 
               'component_index': wmp_idx, 
               'component_human_written': wmp_component, 
               'untrained_components': np.setdiff1d(np.arange(20), [im_idx, wmp_idx, omp_idx])}, 
        
        'OMP':{'session_number':omp_session, 
               'component_index':omp_idx, 
               'component_human_written':omp_component, 
               'untrained_components':np.setdiff1d(np.arange(20), [im_idx, wmp_idx, omp_idx])}}

        if VERBOSE: print(f'{subject_id }, {session_name_mapping}')

        results, subtraction_results = total_variance_change_prepost_learning(subject_id, session_name_mapping)

        results['subject_id'] = subject_id 
        result_list_overall.append(results)

        subtraction_results['subject_id'] = subject_id 
        subtraction_list_overall.append(subtraction_results)

    res_overall = pd.concat(result_list_overall).reset_index(drop=True)
    sub_overall = pd.concat(subtraction_list_overall).reset_index(drop=True)
    res_overall.to_csv(f'{FINAL_RESULTS_PATH}/runwise_neural_variance_control.csv')
    sub_overall.to_csv(f'{FINAL_RESULTS_PATH}/runwise_neural_variance_run_change_control.csv')
    

