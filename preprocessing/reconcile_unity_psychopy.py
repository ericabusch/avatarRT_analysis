## script for reconciling the "psychopy_files" output
# which keeps track of TRs and message passing
# with the output from Unity including directions etc

# runs from command line as
# python reconcile_unity_psychopy.py SUB_ID SES_ID N_RUNS
# then saves to experiment/subjects/SUB_ID/ses-SES_NUM/events_master.csv

# this is to be run after git push/pull from Linux computer

import numpy as np
import pandas as pd
import os, sys, glob, argparse
from sklearn.preprocessing import MinMaxScaler

################ Parse CL args #######################
parser = argparse.ArgumentParser()
parser.add_argument("-sub", "--subject_id",type=str, help="avatarRT_sub_XX")
parser.add_argument("-ses", "--session", type=int, help="session number")
parser.add_argument("-nr", "--num_runs", type=int, help="number of runs")
parser.add_argument("-tr", "--tr_sec",type=int, default=2,help="length of TR in seconds")
parser.add_argument("-x", '--exclude_run',type=int,default=0)
args = parser.parse_args()
SUB_ID = args.subject_id
SES_NUM = args.session
N_RUNS = args.num_runs
TR_in_sec = args.tr_sec
print(f' --> Running {SUB_ID} ses {SES_NUM} for {N_RUNS} runs and TR of {TR_in_sec}')
################ done parsing CL args #######################

if TR_in_sec == None:
    TR_in_sec = 2
else:
    TR_in_sec = int(TR_in_sec)

BOLD_shift = 3 # shift everything 3TR/6s to account for BOLD signal

path = f'../experiment/subjects/{SUB_ID}/ses_{SES_NUM:02d}/behav/'
psychopy_files = sorted(glob.glob(path+'*/psychopy_files/*events*'))[:N_RUNS]
print(path)
# get the number of TRs of fMRI data collected for each run
timing_file = f'../experiment/subjects/{SUB_ID}/ses_{SES_NUM:02d}/scripts/timing.txt'
try:
    with open(timing_file, 'r') as f:
        X = f.readlines()
    fMRI_TRs = [int(x.strip()) for x in X]
    print(fMRI_TRs)
except:
    print(f"Could not load timing file at {timing_file}; getting from data.")
    fMRI_TRs  = []
    for r in range(1,N_RUNS+1):
        x = np.load(f'../experiment/subjects/{SUB_ID}/ses_{SES_NUM:02d}/data/{SUB_ID}_run_{r:02d}_masked_data.npy')
        fMRI_TRs.append(x.shape[0])
            
    print(fMRI_TRs)
        

df_list = []
included_runs = np.arange(1,args.num_runs+1)
if args.exclude_run != 0: included_runs = [i for i in included_runs if i != args.exclude_run]

print("Searching for runs ",included_runs)
# loop through the psychopy files (one per run)
for r_i, run, f in zip(np.arange(len(included_runs)), included_runs, psychopy_files):
    print(f'Looping through: {r_i},{run},{f}')
    N_TRs = fMRI_TRs[r_i]
    print(run, f.split('/')[-1], N_TRs)
    df = pd.read_csv(f, index_col=0)
    # sort by trigger count
    df=df.sort_values(by=['TriggerCount'])
    # Store the round number, position, average rotion, the mode (joystick) 
    # the difficulty of a round, if a TR is inside a round (meaning, not rest between rounds) or if it's a calibration TR (which happens at the beginning)
    rounds, positions, rotations, mode, difficulty, is_round, is_calib = [], [], [], [], [], [], []
    round_counter, state = -1, -1

    # loop through rows of the psychopy files
    for i in range(len(df)):
        row = df.iloc[i]

        # figure out what round we're in
        # and TRs that are between rounds (after end but before begin) are given -1
        
        joycondition = ('Begin' in row.EventValue)
        scancondition = ('Begin' in row.EventValue or 'Start' in row.EventValue)
        conditions = {1:joycondition, 2:scancondition, 3:scancondition, 4:scancondition,5:scancondition}
        if conditions[SES_NUM]:
            round_counter += 1
            state = 0
            rounds.append(round_counter)
            is_round.append(1)
            is_calib.append(0)
        elif "End" in row.EventValue:
            state = -1
            rounds.append(state)
            is_round.append(0)
            is_calib.append(0)
        elif "Calibration" in row.EventValue:
            is_calib.append(1)
            rounds.append(-1)
            is_round.append(0)
        else:
            is_calib.append(0)
            if state == -1:
                rounds.append(state)
                is_round.append(0)
            else:
                rounds.append(round_counter)
                is_round.append(1)

        # if we're within a round, read in the file and get some basic stats about it
        if state != -1:
            try:
                player_info = pd.read_csv(f'{path}/run_{run:03d}/round_{round_counter:02d}/player_info.txt')
            except:
                print(f'{path}/run_{run:03d}/round_{round_counter:02d}/player_info.txt DNE') 
            mode.append(player_info.MoveMode[0])
            difficulty.append(player_info.Difficulty[0])
        else:
            mode.append(np.nan)
            difficulty.append(np.nan)
    # Also load in the timing file, to know how many TRs of fMRI data were collected
    # so we can shift our data accordingly!
    
    
    # Save information
    df['round'] = rounds
    df['is_round'] = is_round
    df['is_calibration'] = is_calib
    df['run'] = [run] * len(rounds)
    df['mode'] = mode
    df['difficulty'] = difficulty
    df.index = df.TriggerCount
    df.drop_duplicates(subset='TriggerCount', inplace=True)
    df['round'] = df['round'].astype(int)
    # now make sure it has enough rows to add
    if df.shape[0] < N_TRs:
        to_add = N_TRs - df.shape[0]
        print(f'padding with {to_add} TRs')
        temp = pd.DataFrame({col: [-1]*to_add for col in df.columns})
        temp.index = np.arange(df.shape[0], df.shape[0]+to_add)
        df = pd.concat([df, temp])
        print("Finished padding")
    df['round_shifted'] = ([-1]*BOLD_shift + list(df['round'].values))[:len(df)] # 
    df['round_shifted'] = df['round_shifted'].astype(int)

    round_shifted = []
    RC = 0
    for i in range(len(df)):
        row = df.iloc[i]
        if row['is_calibration'] == 1:
            round_shifted.append(-1)
        else:
            if (row['round'] != RC) and (row['round'] != -1):
                RC = row['round']
            round_shifted.append(RC)
    #print(round_shifted)
    df['round_shifted'] = round_shifted
    df['is_round_shifted'] = ([np.nan]*BOLD_shift + is_round)[:len(round_shifted)]
    
    for col in ['x', 'z','x_shifted', 'z_shifted','x_norm', 'z_norm', 'z_norm_shifted', 'x_norm_shifted']:
        df[col] = np.repeat(np.nan, len(df))
    
    for rnd in df['round'].unique():
        if rnd < 0:
            continue
        # load in the file with all the info for this round
        try:
            RoundFile = pd.read_csv(f'{path}/run_{run:03d}/round_{rnd:02d}/player_transform.txt')
        except:
            print(f'{path}/run_{run:03d}/round_{rnd:02d}/player_transform.txt DNE')
            continue
        RoundFile = RoundFile[RoundFile['Event'] != 'Event']
        RoundFile['x_norm'] = MinMaxScaler(feature_range=(-1,1)).fit_transform(RoundFile['x'].values.reshape(-1,1))[:,0]
        RoundFile['z_norm'] = MinMaxScaler(feature_range=(-1,1)).fit_transform(RoundFile['z'].values.reshape(-1,1))[:,0]
        
        # figure out what TRs are in the round
        TRs_in_round = df[(df['run'] == run) & (df['round'] == rnd)]['TriggerCount'].unique()
        # if run == 2:
        #     print(df[(df['run'] == run) & (df['round'] == rnd)])
        # print(f'TRs: {TRs_in_round}')
        n_TRs = int(TRs_in_round.max() - TRs_in_round.min())
        if n_TRs == 0:
            print(f'Found {n_TRs} fmri TRs in run {run} round {rnd}; skipping round')
            continue
        # figure out exactly how to bin the data
        bin_size = len(RoundFile) / n_TRs
        
        for b, j in enumerate(TRs_in_round[:-1]):
            idx = np.arange(b * bin_size, b * bin_size + bin_size).astype(int)
            idx = idx[idx < len(RoundFile)]
            xvals = [float(n) for n in RoundFile['x'].values[idx]]
            zvals = [float(n) for n in RoundFile['z'].values[idx]]
            xnorm_vals = [float(n) for n in RoundFile['x_norm'].values[idx]]
            znorm_vals = [float(n) for n in RoundFile['z_norm'].values[idx]]
            
            TR_idx = df[df['TriggerCount'] == j].index
            for IDX in TR_idx:
                df.at[IDX, 'x'] = np.mean(xvals)
                df.at[IDX, 'z'] = np.mean(zvals)
                df.at[IDX + BOLD_shift, 'x_shifted']= np.mean(xvals)
                df.at[IDX + BOLD_shift, 'z_shifted']= np.mean(zvals)
                df.at[IDX, 'x_norm'] = np.mean(xnorm_vals)
                df.at[IDX, 'z_norm'] = np.mean(znorm_vals)
                df.at[IDX + BOLD_shift, 'x_norm_shifted']= np.mean(xnorm_vals)
                df.at[IDX + BOLD_shift, 'z_norm_shifted']= np.mean(znorm_vals)
                
    
    df['data_TRs-0idx'] = df['TriggerCount'].values - 1
    df.to_csv(f"./{path}/run_{run:03d}_events_master.csv")
    
    print(f'saved to ./{path}/run_{run:03d}_events_master.csv')
    if df['is_calibration'].sum() > 0:
        c1 = df['TriggerCount'] >= 10
        print('retaining non-calibration trs')
    else:
        c1 = df['TriggerCount'] >= 0
    c2 = df['is_calibration'] != 1
    c3 = df['Event'] == 'Trigger'
    c4 = df['EventValue'] == 'Start_TR_1'

    only_TRs = df[c1& c2& c3]
    only_TRs = pd.concat([only_TRs, df[c4]])
    cols = only_TRs.columns
    cols = [c for c in cols if c not in ['index','TriggerCount.1','level_0']]
    only_TRs=only_TRs[cols]

    outfn = f'./{path}/run_{run:03d}_events_TRs.csv'
    only_TRs.to_csv(outfn)
    
    # which TRs to include in analyses?
    include_TRs = only_TRs[only_TRs['is_round_shifted'] == 1]['data_TRs-0idx'].values
    np.save(f'./{path}/run_{run:03d}_round_TRs_0idx_label_shifted.npy', include_TRs)
    
    df_list.append(df)

psy_df = pd.concat(df_list)

prev=-1
runner=0
relabeled=[]
for value in psy_df.TriggerCount.values:
    if value < prev:
        runner += prev
    currentTR = runner+value
    relabeled.append(currentTR)
    prev=value

psy_df['data_TRs-1idx']=relabeled
psy_df['data_TRs-0idx']=[r-1 for r in relabeled]

outfn = f'./{path}/events_master.csv'
psy_df.to_csv(outfn)
print(f'saved to {outfn}')

# now take this last file, and retain only the information about where there are triggers AFTER calibration
if psy_df['is_calibration'].sum() > 0:
    c1 = psy_df['TriggerCount'] >= 10
    print('retaining non-calibration trs')
else:
    c1 = psy_df['TriggerCount'] >= 0
c2 = psy_df['is_calibration'] != 1
c3 = psy_df['Event'] == 'Trigger'
c4 = psy_df['EventValue'] == 'Start_TR_1'

only_TRs = psy_df[c1& c2& c3]
only_TRs = pd.concat([only_TRs, psy_df[c4]])

outfn = f'./{path}/events_TRs.csv'
only_TRs.to_csv(outfn)

print(f'saved to {outfn}')



