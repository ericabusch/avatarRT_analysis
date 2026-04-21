import numpy as np
from brainiak.utils import fmrisim
from brainiak.utils import fmrisim_real_time_generator
import logging, shutil
import os, sys, glob, argparse, warnings
sys.path.append(os.getcwd())
import nibabel as nib
from nibabel import Nifti1Image
import json, random
import time, datetime
import pydicom as dicom

def my_write_dicom(output_name,
                 data,
                 image_number=0, RepTime=2):
    """Write the data to a dicom file
    Saves the data for one TR to a dicom.

    Dicom files are difficult to set up correctly, this file will likely
    crash when trying to open it using dcm2nii. However, if it is loaded in
    python (e.g., dicom.dcmread) then pixel_array contains the relevant
    voxel data

    Parameters
    ----------

    output_name : str
        Output name for volume being created

    data : 3 dimensional array
        Volume of data to be saved

    image_number : int
        Number dicom to be saved. This is critical for setting up dicom file
        header information.

    """

    # Convert data from float to in
    dataInts = data.astype(np.int16)

    # Populate required values for file meta information
    file_meta = dicom.Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2'  # '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'

    # Create the FileDataset
    ds = dicom.FileDataset(output_name,
                           {},
                           file_meta=file_meta,
                           preamble=b"\0" * 128)

    # Set image dimensions
    frames, rows, cols = dataInts.shape
    ds.Rows = rows
    ds.Columns = cols
    ds.NumberOfFrames = frames
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.PixelRepresentation = 0
    ds.InstanceNumber = image_number
    ds.ImagePositionPatient = [0, 0, 0]
    ds.ImageOrientationPatient = [.01, 0, 0, 0, 0, 0]
    ds.PhotometricInterpretation = 'MONOCHROME1'

    # Add the data elements -- not trying to set all required here. Check DICOM
    # standard
    ds.PatientName = "sim"
    ds.PatientID = "sim"
    ds.RepetitionTime = RepTime

    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    script_datetime = datetime.datetime.now()
    # Set creation date/time
    image_datetime = script_datetime + datetime.timedelta(seconds=image_number)
    timeStr = image_datetime.strftime('%H%M%S')
    ds.ContentDate = image_datetime.strftime('%Y%m%d')
    ds.ContentTime = timeStr

    # Add the data
    ds.PixelData = dataInts.tobytes()

    ds.save_as(output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nTRs', '-n', default=300, type=int, help='number of volumes to simulate')
    parser.add_argument('--sub','-s',default='XX',type=str,help='simulated subject id')
    parser.add_argument('--ses','-e',default='02',type=str,help='simulated session id')
    parser.add_argument('--run','-r',default=1,type=int,help='simulated run id')
    parser.add_argument('--chosen','-c',default=5,type=int, help='which real subject is the basis for this simulation?')
    parser.add_argument('--outdir', '-o', default='', type=str,
                           help='where to save data')
    parser.add_argument('--file_format', '-f', default="001_{X:06d}_{TR:06d}.dcm", 
                           help='')
    parser.add_argument('--TR', '-t', default=2, type=float,
                           help='length of TR in seconds')
    parser.add_argument("--real_wait", '-re', default=True, type=bool)
    parser.add_argument('--verbose','-v',default=True,type=bool)
    args = parser.parse_args()
    path = "./rt-cloud/projects/avatarRT/"
    if len(args.outdir) == 0:
        args.outdir = f'{path}/experiment/subjects/avatarRT_sub_{args.sub}/avatarRT_sub_{args.sub}_ses_{args.ses}'
    os.makedirs(args.outdir, exist_ok=True)
    if args.verbose: print(f'saving to {args.outdir}')
    ref_dir = f'{path}/experiment/subjects/avatarRT_sub_{args.sub}/reference'
    os.makedirs(ref_dir, exist_ok=True)
    
    save_dicom=False
    if args.file_format.split('.')[-1] == 'dcm':
        save_dicom = True
    else:
        save_dicom = False
    verbose = args.verbose
    
    chosen_subject=args.chosen
    
    vol = f"{path}/experiment/subjects/avatarRT_sub_{chosen_subject:02d}/ses_01/func/avatarRT_sub_{chosen_subject:02d}_task-joystick_run-01_bold.nii"
    template_path = f"{ref_dir}/sample_template.npy"
    mask_path = f"{ref_dir}/sample_mask.npy"  
    
    if not os.path.exists(template_path):
        mask, template = fmrisim.mask_brain(volume=nib.load(vol).get_fdata(), mask_self=True)
        np.save(mask_path, mask)
        np.save(template_path, template)
        if verbose: print(f"saved {template_path}, {mask_path}, vol={np.sum(nib.load(vol).get_fdata())}")
    
    template,mask = np.load(template_path), np.load(mask_path)
    if verbose: print(f"loaded {template_path}, {mask_path}")

    # estimate noise
    if not os.path.exists(f'{ref_dir}/noise_dict.txt'):
        noise_dict = fmrisim.calc_noise(nib.load(vol).get_fdata(), mask, template)
        with open(f'{ref_dir}/noise_dict.txt','w') as f:
            f.write(json.dumps(noise_dict))
    # else:
    #noise_dict = json.load(f'{ref_dir}/noise_dict.txt')
    with open(f'{ref_dir}/noise_dict.txt','r') as f:
        noise_dict = json.load(f)#.read()
    # noise_dict = eval(noise_dict)
    noise_dict['matched'] = 0
    dimensions = nib.load(vol).shape[:3]
    
    # generate noise
    if not os.path.exists(f'{args.outdir}/noise_vol.npy'):
        noise = fmrisim.generate_noise(dimensions=dimensions,
                               stimfunction_tr=np.zeros((args.nTRs, 1)),
                           tr_duration=args.TR,
                           template=template,
                           mask=mask,
                           noise_dict=noise_dict,
                           )
        np.save(f'{args.outdir}/noise_vol.npy', noise)
        print("Saved noise")


    noise = np.load(f'{args.outdir}/noise_vol.npy')

    # choose a random subject's mask
    roi_mask_to_copy = f'{path}/experiment/subjects/avatarRT_sub_{chosen_subject:02d}/reference/mask.nii.gz'
    if verbose: print(f'copying {roi_mask_to_copy}')
    
    if verbose: print("Generated noise volume; starting run")
    for idx in range(args.nTRs):
        
        brain = noise[...,idx]
        # if verbose: print(f'sum of brain: {np.sum(brain)}')
        # Convert file to integers to mimic what you get from MR
        brain_int32 = np.squeeze(brain.astype(np.int32))
        out_fn = os.path.join(args.outdir, args.file_format.format(X=args.run, TR=idx))
        
        if save_dicom:
            my_write_dicom(out_fn, brain_int32, idx+1, RepTime=2)
        else:
            np.save(out_fn, brain_int32)
        if verbose: print(f"TR {idx} saved {out_fn.split('/')[-1]} shape {brain_int32.shape}")
        #if args.real_wait: time.sleep(args.TR)

    print("Done real-time")
    if args.ses == '01':
        outfn = f'{path}/experiment/subjects/avatarRT_sub_{args.sub}/reference/mask.nii.gz'
        roi_mask_orig = nib.load(roi_mask_to_copy)
        print(f'extracting ROI')
        this_mask = mask.copy()
        new_mask = np.multiply(this_mask, roi_mask_orig.get_fdata())
        new_nii = Nifti1Image(new_mask, affine=roi_mask_orig.affine)
        # select voxels
        nib.save(new_nii,outfn)
        print(f'saved to {outfn}')
        coords=np.where(new_mask == 1)
        x,y,z=coords
        print(f'n voxels in mask: {len(coords[0])}')
        masked_data = noise[x,y,z,:] 
        print(f'saving masked data of shape {masked_data.shape}')
        np.save(f'{args.outdir}/brain_masked_voxels.npy', masked_data) 
        np.save(f'{args.outdir}/brain_volume_sim.npy', noise)
        outfn = f'{ref_dir}/ses_01_example_func.nii.gz'
        new_nii = Nifti1Image(brain, affine=roi_mask_orig.affine)
        nib.save(new_nii, outfn)
        print(f'saved {outfn}')      
        os.makedirs(f'{path}/SCANS', exist_ok=True)
        fns = glob.glob(f'{path}/*dcm')
        for f in fns:
            shutil.move(f, f.replace(f'{path}', f'{path}/SCANS/'))

