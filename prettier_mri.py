#!/usr/bin/env python3

# Diana Giraldo, Jan 2024
# Last update: April 2024

import os, sys
from pathlib import Path
import argparse
import torch

import numpy as np
import nibabel as nib

repo_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, repo_dir)

from prettier.srr_mri import reconstruct_volume
from prettier.utils.affine_transform import adjust_affine_transform

EPS_INPLANE = 0.25
COMBINE_VOL_METHOD = "average"
FT_WEIGHTS_URLS = {
    'RealESRGAN_v1': "https://drive.google.com/uc?export=download&id=15xWVa7C4IISiMlXIdee2yjjZne2dufJh",
    'EDSR_v1': "https://drive.google.com/uc?export=download&id=13E-EKIdHW6QyrZiLE8WvvDcJ1vnP9RgS",
    'ShuffleMixer_v1': "https://drive.google.com/uc?export=download&id=1sg2P2SNIW-efGflzYCHlWcRUuzjsSYTd"
}

# -----------------------------------------
# --- Perform SRR with fine-tuned models ---
# -----------------------------------------

def main(args=None):
    
    # Get inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, 
    help = 'input image')
    parser.add_argument('-o', '--output', type=str, required=True, 
    help = 'output image')
    parser.add_argument('-m', '--model-name', type=str, choices=["EDSR", "RealESRGAN", "ShuffleMixer"], required=True, 
    help = 'select fine-tuned model')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--no-flip-axes', action='store_true', default=False)
    parser.add_argument('--keep-in-plane', action='store_true', default=False,
    help = 'keep original in-plane resolution. Outputs could have distortions when in-plane resolution is <<< 1mm')
    parser.add_argument('--intermediate-volumes', action='store_true', default=False)
    parser.add_argument('--version', type=int, default=1, choices=[1,2])
    parser.add_argument('--quiet', action='store_true', default=False)
    
    args = parser.parse_args(args if args is not None else sys.argv[1:])
    model_name = args.model_name
    print_info = not args.quiet
    use_version = args.version
    adjust_in_plane = not args.keep_in_plane

    interpolate_lr = False if use_version == 1 else True
    mod_up_scale = 4 if use_version == 1 else 1
    
    if print_info: 
        print('-------------------------------------------------')
        print('If you use PRETTIER MRI, please cite:')
        print('Giraldo, et al. (2024). Perceptual super-resolution in multiple sclerosis MRI.')
        print('Frontiers in Neuroscience (Vol. 18). https://doi.org/10.3389/fnins.2024.1473132')
        print('-------------------------------------------------')
    
    # Check inputs
    if not os.path.isfile(args.input):
        raise ValueError('Input image does not exist.')
        
    # if not os.path.isdir(os.path.dirname(args.output)):
    #     raise ValueError('Output directory does not exist.')
    
    # Check existence of weights directory
    weights_dir = os.path.join(repo_dir, "weights")
    if not os.path.isdir(weights_dir):
        raise RuntimeError(f'Weights directory not found in {weights_dir}')
    
    # Check existence of fine-tuned weights, try to download it 
    if use_version == 1:
        weights_fname = f"{model_name}_finetuned.pth"
    else:
        weights_fname = f"{model_name}_finetuned_v{use_version}.pth"

    weights_fpath = os.path.join(weights_dir, weights_fname)

    if not os.path.isfile(weights_fpath):
        #raise ValueError('File with weights not found in', weights_fpath)
        if print_info: print(f'Fine-tuned weights not found in {weights_fpath}, trying to download it...')
        import gdown
        gdown.download(FT_WEIGHTS_URLS[f"{model_name}_v{use_version}"], weights_fpath, quiet=args.quiet)

    else: 
        if print_info: print(f'Fine-tuned weights found in {weights_fpath}')
        
    # Detect device: GPU? MPS? -> CPU
    if torch.cuda.is_available():
        dev_str = f'cuda:{args.gpu_id}'
    elif torch.backends.mps.is_available():
        dev_str = 'mps'
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    if print_info: print("Device:", device)
    
    if print_info: print('-------------------------------------------------')

    ###############################    
    # Read and adjust input image
    LR = nib.load(args.input)
    
    if print_info: 
        print("LR image:", args.input)
        print('Image array shape:', LR.header['dim'][1:4])
        print('Voxel size:', LR.header['pixdim'][1:4])
        print('Orientation of voxel axes:', nib.aff2axcodes(LR.affine))
    
    original_ori = nib.io_orientation(LR.affine)
    flip_axes = (np.min(original_ori[:,1]) < 0) and not args.no_flip_axes
    
    if flip_axes:
        if print_info: print("Flipping axes of LR input")
        flipping = original_ori
        flipping[:,0] = [0,1,2]
        if print_info: print(flipping)
        LR = LR.as_reoriented(flipping)
        if print_info: print('New orientation of voxel axes:', nib.aff2axcodes(LR.affine))

    LR_voxelsize = np.array(LR.header.get_zooms())
    LR_v2w = LR.affine.astype(np.float64)
    if adjust_in_plane and abs(LR_voxelsize[0] - 1) > EPS_INPLANE:
        import nilearn.image
        if print_info: print('Adjusting in-plane resolution closer to 1mm')
        ip_factor = round(1/LR_voxelsize[0])
        ip_adjust = np.array([1/ip_factor, 1/ip_factor, 1])
        new_LR_shape = (np.array(LR.shape)*ip_adjust).astype(int)
        new_LR_v2w = adjust_affine_transform(LR_v2w, ip_adjust)
        newLR = nilearn.image.resample_img(
            LR, 
            target_affine=new_LR_v2w, 
            target_shape=tuple(new_LR_shape),
            force_resample=True,
            copy_header=True,
        )  
        LR = newLR   
        if print_info: 
            print('New shape:', LR.header['dim'][1:4])
            print('New voxel size:', LR.header['pixdim'][1:4])   

    # Get trained models
    if print_info:     
        print('-------------------------------------------------')
        print(f'Loading trained {model_name} model')
    
    # Get model
    if model_name == "EDSR":
        from prettier.models.edsr import EDSR
        model = EDSR(n_colorchannels = 3, scale = mod_up_scale)
        model.load_state_dict(torch.load(weights_fpath, map_location=device)['model_weights'])
        scale_factor = 255
        
    elif model_name == "RealESRGAN":
        from basicsr.archs.rrdbnet_arch import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale= mod_up_scale)
        model.load_state_dict(torch.load(weights_fpath, map_location=device)['generator_weights'])
        scale_factor = None
    
    elif model_name == "ShuffleMixer":
        from prettier.models.shufflemixer import ShuffleMixer
        model = ShuffleMixer(n_feats=64, kernel_size=7, n_blocks=5, mlp_ratio=2, upscaling_factor = mod_up_scale)
        model.load_state_dict(torch.load(weights_fpath, map_location=device)['model_weights'])
        scale_factor = None
        
    else:
        print("model name not valid")
    
    model.eval()
    model = model.to(device)
    
    # Reconstruct volume
    rec, v2w, *vols = reconstruct_volume(
        LR, 
        model, 
        device,
        batch_size = args.batch_size,
        scale_factor = scale_factor,
        slices_as_channels = True,
        select_middle = False,
        interpolate_lr = interpolate_lr,
        return_vol_list = args.intermediate_volumes,
        combine_vol_method = COMBINE_VOL_METHOD,
        print_info = print_info,
    )
    
    hr_nib = nib.Nifti1Image(rec, v2w)

    if args.intermediate_volumes:
        rec_nibs = [nib.Nifti1Image(vol, v2w) for vol in vols[0]]
    
    # Flip back axes
    if flip_axes:
        #if print_info: print('Orientation of voxel axes:', nib.aff2axcodes(hr_nib.affine))
        if print_info: print("Flipping axes of HR output")
        hr_nib = hr_nib.as_reoriented(flipping)
        #if print_info: print('New orientation of voxel axes:', nib.aff2axcodes(hr_nib.affine))
        if args.intermediate_volumes:
            rec_nibs = [vol_nib.as_reoriented(flipping) for vol_nib in rec_nibs] 
            
    # Save nifti image
    nib.save(hr_nib, args.output) 
    if print_info: print("Reconstructed image saved in", args.output)

    if args.intermediate_volumes:
        suff = ''.join(Path(args.output).suffixes)
        stem = args.output.rstrip(suff)
        for i, vol_nib in enumerate(rec_nibs):
            tmp_out = stem + '_vol' + str(i) + suff
            nib.save(vol_nib, tmp_out) 
            if print_info: print("Intermediate volume saved in", tmp_out)


if __name__ == '__main__':
    main()