import warnings
warnings.filterwarnings("ignore")
import numpy as np
import sys
import pandas as pd
sys.path.append("..")
import avatarRT_utils
from MRAE.mrae import ManifoldRegularizedAutoencoder
from MRAE import dataHandler
import matplotlib.pyplot as plt
import scipy, os, json
from TPHATE.tphate import tphate
from scipy.stats import zscore, percentileofscore, ttest_ind
import scprep, glob, argparse
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from subprocess import call
import nibabel as nib

def mask_func_data_all(indir, outdir, mask_fn, crop=None):
    files = sorted(glob.glob(f'{indir}/*preproc.nii.gz'))
    file_root = [f.split("/")[-1].replace('bold_preproc.nii.gz', 'navigation.npy') for f in files]
    mask = nib.load(mask_fn).get_fdata()
    masked_data = []
    for i, fn in enumerate(files):
        data = nib.load(fn).get_fdata()
        data_masked = data[mask!=0].T
        data_masked = np.nan_to_num(avatarRT_utils.normalize(data_masked, axis=0))
        outfn = file_root[i]
        if crop: data_masked = data_masked[crop:]
        if verbose: print(f'data of shape: {data_masked.shape} to {os.path.join(outdir, outfn)}')
        # save to file
        np.save(os.path.join(outdir, outfn), data_masked)
        masked_data.append(data_masked)
    return masked_data

def mask_func_data_rounds(indir, outdir, mask_fn, behav_dir):
    files = sorted(glob.glob(f'{indir}/*preproc.nii.gz'))
    file_root = [f.split("/")[-1].replace('bold_preproc.nii.gz', 'navigation_round_TRs.npy') for f in files]
    
    mask = nib.load(mask_fn).get_fdata()
    masked_data = []
    for i, fn in enumerate(files): 
        run = i+1
        data = nib.load(fn).get_fdata()
        fn = f"{behav_dir}/run_{run:03d}_round_TRs_0idx_label_shifted.npy"
                              
        TR_arr = np.load(fn)
        print(len(TR_arr), np.shape(data))
        
        last_TR = np.min([data.shape[-1], np.max(TR_arr)])
        print(f"TRs in data: {data.shape[-1]}; max TRs in file: {np.max(TR_arr)}; taking min={last_TR}")
        TR_arr = TR_arr[TR_arr < last_TR].astype(int)
        data_masked = data[mask!=0].T
        data_masked = np.nan_to_num(avatarRT_utils.normalize(data_masked, axis=0))
        data_masked_trimmed = data_masked[TR_arr]
        outfn = file_root[i]
        print(f"data of shape {data_masked_trimmed.shape} to {os.path.join(outdir, outfn)}")
        np.save(os.path.join(outdir, outfn), data_masked)
        masked_data.append(data_masked)
    return masked_data

def make_embeddings_visualize(X, run_num, outdir, isMasked=False):
    embed20 = avatarRT_utils.embed_tphate(X, n_components=20)
    endstr='_masked_TRs' if isMasked else ''
    np.save(f'{outdir}/{sub_id}_run-{run_num:02d}_navigation_embedding_20d{endstr}.npy', embed20)
    embed = avatarRT_utils.embed_tphate(X, t=5, n_components=2)
    scprep.plot.scatter2d(embed, c=np.arange(embed.shape[0]), 
                          filename=f'{outdir}/plots/{sub_id}_run-{run_num:02d}_tphate{endstr}_time_color.png', ticks=False)
    return embed20

def visualize_additional_embeddings(X, run_num, outdir, behav_dir, isMasked=False):
    label_df = pd.read_csv(f'{behav_dir}/run_{run_num:03d}_events_TRs.csv')
    endstrhere='_masked_TRs' if isMasked else ''
    to_label = ['round_shifted', 'x_shifted', 'z_shifted', 'x_norm_shifted','z_norm_shifted']
    if not isMasked:
        to_label += ['is_round']
        max_TR = label_df['data_TRs-0idx'].values.max()
        max_TR = np.min([X.shape[0], max_TR])
        TRs_to_include = np.arange(max_TR)
    else:
        TRs_to_include = np.load(f"{behav_dir}/run_{run_num:03d}_round_TRs_0idx_label_shifted.npy")
        max_TR = np.min([X.shape[0], np.max(TRs_to_include)])
        TRs_to_include = TRs_to_include[TRs_to_include < max_TR]
        # get the same mapping back to how the data was broken up 
    
    # subsample the label df accordingly
    TRs_to_include = TRs_to_include.astype(int)
    X = X[TRs_to_include]
    label_df = label_df[label_df['data_TRs-0idx'].isin(TRs_to_include)][to_label]

    print(f'Label shape: {label_df.shape}, X shape: {X.shape}')
    
    assert len(label_df) == len(X)
    embed = avatarRT_utils.embed_tphate(X, t=5, n_components=2)
    print(f"Made embedding shape: {embed.shape}")
    for label_set in to_label:
        endstr = f'{endstrhere}_{label_set}'
        title = f"Run {run_num} by {label_set} "
        if isMasked:
            title += ' masked TRs' 
        lab = label_df[label_set].values
        scprep.plot.scatter2d(embed, c=lab, title=title,
                          filename=f'{outdir}/plots/{sub_id}_run-{run_num:02d}_tphate{endstr}.png', 
                              ticks=False)
        print("finished ",title)


def run_MRAE(all_data, train_percent, model_outdir, train_epochs=10000, verbose=True):
    n_timepoints, IO_dim = all_data.shape
    if verbose: print(f'input data of shape {all_data.shape}')
    n_train_timepoints = n_timepoints * train_percent // 100
    train_timepoints = np.arange(n_timepoints)[:n_train_timepoints]
    train_data = all_data[train_timepoints]
    test_timepoints = np.setdiff1d(np.arange(n_timepoints), train_timepoints)
    test_data = all_data[test_timepoints]
    
    np.save(model_outdir+'/model_train_data.npy', train_data)
    np.save(model_outdir+'/model_test_data.npy', test_data)
    
    tphate_op = tphate.TPHATE(verbose=0, n_components=20, t=5)
    trainDataset = dataHandler.BuildVolumeAndEmbedDataset_single(data=train_data, EMBED_OP=tphate_op)
    
    mrae = ManifoldRegularizedAutoencoder(64, manifold_dim=trainDataset.get_embed_dims(), 
                                          IO_dim=trainDataset.get_voxel_dims()[0], 
                                          n_epochs=train_epochs, ckp_epoch=100)
    
    params = {'hidden_dim':mrae.hidden_dim, 'manifold_dim':mrae.manifold_dim, 
              'IO_dim': trainDataset.get_voxel_dims()[0], 'n_epochs':mrae.n_epochs, 
              'learning_rate':mrae.lr, 'batch_size':mrae.batch_size, 'seed':mrae.seed, 
             'lambda_embed':mrae.lambda_embd}
    # dump specs 
    with open(model_outdir+'/modelSpec.txt','w') as f:
        f.write(json.dumps(params))
    bottleneck = trainDataset.embeds
    np.save(model_outdir+'/bottleneck.npy', bottleneck)
    if verbose: print(f'saved model spec and bottleneck to {model_outdir}; starting training')
    os.makedirs(f'{model_outdir}/checkpoints/', exist_ok=True)
    overall_losses, embed_loss = mrae.train(trainDataset, save_dict_path=f'{model_outdir}/state_dict.pt',
                       ckp_path=f'{model_outdir}/checkpoints/', verbose=True)
    # plot loss
    fig,ax=plt.subplots(figsize=(4,4))
    ax.plot(overall_losses)
    ax.set(title='training loss', xlabel='epoch', ylabel='reconstruction MSE')
    plt.savefig(f'{model_outdir}/training_loss.png')
    np.save(f'{model_outdir}/training_loss.npy', overall_losses)
    np.save(f'{model_outdir}/embedding_loss.npy',embed_loss)
    # embed train data
    proj_train = avatarRT_utils.project_new_data(mrae, train_data)
    
    # run PCA - get all the PCs
    components = avatarRT_utils.get_manifold_component(proj_train)
    
    if verbose: print("extracting test data")
    test_projection = avatarRT_utils.project_new_data(mrae, test_data)
    saveto = f'{model_outdir}/projected_test_runs.npy'
    np.save(saveto, test_projection)
    # for all PCs, save them, project test data, and extract range
    for i in range(components.shape[0]):
        pc = components[i,:]
        np.save(model_outdir+f'/manifold_pc_{i+1:02d}.npy', pc)
        mapped_test_data = avatarRT_utils.map_projection(test_projection, pc)
        testRange = np.percentile(mapped_test_data, [1,99])
        np.save(model_outdir+f'/test_range_{i+1:02d}.npy', testRange)
        print(f'Test range on pc {i+1:02d} : {testRange}')
    

# def determine_perturbations(all_data, model_outdir):
#     # read in the old data
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-sub", "--subject_id",type=str, help="avatarRT_sub_XX")
    parser.add_argument("-ses", "--session", type=int, help="session number", default=0)
    parser.add_argument("-mtr", "--mask_TRs", type=int, help="Do you want to only embed the data that are in runs?", default=1)
    parser.add_argument("-aev", "--additional_embedding_vis", type=int, default=0, help="Do you want to visualize additional embeddings?")
    args = parser.parse_args()
    sub_id = args.subject_id
    ses_id = f'ses_{args.session:02d}'
    mask_TRs = args.mask_TRs # do we want to only embed the round data?
    additional_embedding_vis = args.additional_embedding_vis # do we want to vis embeddings with labels from game variables
    
    verbose = True
    experiment_dir = f'../experiment/subjects/{sub_id}/{ses_id}'
    func_dir = os.path.join(experiment_dir, 'func')
    out_dir = os.path.join(experiment_dir, 'ROI_data', 'data')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.makedirs(out_dir+'/plots')
        
    mask_fn = os.path.join(f'../experiment/subjects/{sub_id}', 'reference', 'mask.nii.gz')
    behav_dir = os.path.join(experiment_dir, 'behav_raw')

    crop=10    
    if mask_TRs == 0:    
        masked_data = mask_func_data_all(func_dir, out_dir, mask_fn, crop=crop)
    else:
        masked_data = mask_func_data_rounds(func_dir, out_dir, mask_fn, behav_dir)
        
    # embed the data by run for validation
    out_dir = os.path.join(experiment_dir, 'ROI_data', 'embeddings')
    os.makedirs(out_dir+'/plots', exist_ok=True)

    for i, data in enumerate(masked_data):
        make_embeddings_visualize(data, i+1, out_dir, isMasked=mask_TRs)
        if additional_embedding_vis:
            visualize_additional_embeddings(data, i+1, out_dir, behav_dir, isMasked=mask_TRs)
    
    # now train the model
    epochs = 10000
    if mask_TRs:
        model_outdir = f'../experiment/subjects/{sub_id}/model_rerun'
    else:
        model_outdir = f'../experiment/subjects/{sub_id}/model_inc_rest_rerun'
    print(f"Model will be saved to: {model_outdir}")
    os.makedirs(model_outdir, exist_ok=True)
    masked_data_concat = np.concatenate(masked_data, axis=0)
    run_MRAE(masked_data_concat, 80, model_outdir, train_epochs=epochs)
