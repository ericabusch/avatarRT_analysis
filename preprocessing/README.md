# Pipeline for preprocessing day 1 data: 

variables: session_id, username, password (for xnat pull) ; subject_id session_num

1. Activate environment:   
`. ~/run_rtcloud_setup.sh`

2. Get the data from xnat   
`sbatch get_xnat.sh $session_id $username $password`

3. Go from dicom to nifti images   
`sbatch dicom2nii2bids.sh $session_id`

4. Organize the files and rename them accordingly - make sure that the variables in this file match the runs that were collected   
`python offline_prep_ses1.py -sub $subject_id -ses $session_num`
(or for RT processing: `python offline_prep_RT_sessions.py -sub $subject_id -ses $session_num`)

5. Prepare fieldmap images and run bet on them   
`sbatch prep_fieldmap.sh $subject_id ses_0$session_num`

6. Render the feat scripts for this run (final parameter is the number of runs)   
`bash render_feat_scripts.sh $subject_id ses_0$session_num 4`

7. Reorganize files after feat (final parameter is the number of runs)   
`python reorganize_post_feat.py -sub $subject_id -ses $session_num -nr 4`
- this file takes the filtered files after feaat and brings them into one directory, also copies a brain mask and an example func volume (masked) into the reference directory

7. Align the ROI into the session 1 space   
`bash align_ROI.sh $subject_id ses_0$session_num`

8. Reconcile the timing files and the stimulus information. # subject, session, number of runs, TR in sec 
`python reconcile_unity_psychopy.py -sub $subject_id -ses $session_num -nr 4 -tr 2`
- this does automatic TR shifting of the x,y,z coordinates and round number (2 TR shift, 4s)

9. Mask data, make TPHATE embeddings, train MRAE and compute components. 
`python prepare_day1_model.py -sub $subject_id -ses $session_num -mtr 1 -aev 1` # mask TRs within the round, and also run those additional embeddings
