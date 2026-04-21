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

    results = pd.DataFrame(columns=['session_type', 'session_id', 'run','current_NFB_component_idx','congruent', 
        'comparison_component_idx', 'evr', 'comparison_component_name', 'total_var'])
    
    subtraction_results = pd.DataFrame(columns=['session_type','session_id','current_NFB_component_idx', 'congruent', 
        'comparison_component_idx', 'delta_evr_since_perturb', 'delta_evr_overall', 'delta_variance_since_perturb', 'delta_variance_overall', 'comparison_component_name'])
    run_numbers = [1,2,3,4]

    # the nfb component idx is the component index being trained in a given run
    # the comparison component index is the component along which we're measuring change for this row
    # overall means run 1 - run 4, perturb is since the perturbation was introduced
    # comparison component name can be "IM',"WMP','OMP','untrained_{i}'    
    
    for ses_name, ses_info in session_name_mapping.items():
        ses_id = ses_info['session_number']
        ses_id=f'ses_0{ses_id}'
        nfb_component_idx = ses_info['component_index']
        untrained_components = ses_info['untrained_components'] 
        first_run, last_run = SESSION_TYPES_RUNS[ses_name][0], SESSION_TYPES_RUNS[ses_name][-1]
        # each component
        component_names = ['' for i in range(20)]
        for comp_idx in range(20):
            # figure out which component this is 
            comp_name = ''
            if comp_idx in untrained_components:
                comp_name = f'untrained_{comp_idx}'
            elif comp_idx == nfb_component_idx:
                comp_name = ses_name
            else: # it's one of the other two 
                for sn, si in session_name_mapping.items():
                    if comp_idx == si['component_index']:
                        comp_name=sn
            component_names[comp_idx]=comp_name
        print(f'{ses_name}, {component_names}')
        
        for run in run_numbers:
            
            # this will return projected onto the potential NFB comps
            XTarget = helper.load_component_data(subject_id , ses_id, run, component_number=None, by_trial=False, shift_by=2)
            evr_mat = helper.run_EVR(XTarget)
            total_var_mat = helper.compute_total_variance(XTarget)
            
            for comp_idx in range(total_var_mat.shape[-1]):
                comp_name=component_names[comp_idx]        
                results.loc[len(results)] = {'session_type':ses_name, 
                                            'session_id':ses_id, 
                                             'run':run,
                                             'current_NFB_component_idx':nfb_component_idx,
                                            'congruent':nfb_component_idx==comp_idx,
                                            'comparison_component_idx':comp_idx,
                                            'comparison_component_name':comp_name,
                                            'total_var':total_var_mat[comp_idx],
                                            'evr':evr_mat[comp_idx]}
       
        # compare across runs, within this session, shift, etc
        perturb_run, last_run = SESSION_TYPES_RUNS[ses_name][0], SESSION_TYPES_RUNS[ses_name][-1]
        temp = results[results['session_id']==ses_id].reset_index(drop=True)
        for comp_idx in range(20):
            comp_name=component_names[comp_idx] 
            temp = results[(results['comparison_component_idx'] == comp_idx) & (results['session_id']==ses_id)].reset_index(drop=True)
            # get all the info for this component
            pert_run_data= temp[temp['run']==perturb_run]
            last_run_data=temp[temp['run']==last_run]
            delta_evr = last_run_data['evr'].item() - pert_run_data['evr'].item()
            delta_variance =  last_run_data['total_var'].item() - pert_run_data['total_var'].item()
            
            run1_data = temp[temp['run']==1]
            delta_evr_overall = last_run_data['evr'].item() - run1_data['evr'].item()
            delta_variance_overall =  last_run_data['total_var'].item() - run1_data['total_var'].item()
            
            subtraction_results.loc[len(subtraction_results)] = {'session_type':ses_name,
                                                                 'session_id': ses_id,
                                                                 'current_NFB_component_idx':  pert_run_data['current_NFB_component_idx'].item(),
                                                                 'congruent': comp_idx==pert_run_data['current_NFB_component_idx'].item(),
                                                                 'comparison_component_idx': comp_idx,
                                                                 'delta_evr_since_perturb': delta_evr,
                                                                 'delta_evr_overall': delta_evr_overall,
                                                                 'delta_variance_since_perturb': delta_variance,
                                                                 'delta_variance_overall':delta_variance_overall,
                                                                 'comparison_component_name': comp_name}
                        
            
            
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
    res_overall.to_csv(f'{FINAL_RESULTS_PATH}/runwise_component_EVR_neural_analysis_by_trial_control.csv')
    sub_overall.to_csv(f'{FINAL_RESULTS_PATH}/runwise_component_EVR_neural_analysis_run_change_control.csv')
    

