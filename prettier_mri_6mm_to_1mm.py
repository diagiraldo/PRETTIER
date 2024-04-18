#!/usr/bin/env python3

# Diana Giraldo, Jan 2024
# Last update: April 2024

import os, sys
import argparse
import torch

import numpy as np
import nibabel as nib

repo_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, repo_dir)

from prettier.srr_mri import reconstruct_volume

COMBINE_VOL_METHOD = "average"
FT_WEIGHTS_URLS = {
    'RealESRGAN': "https://drive.google.com/uc?export=download&id=15xWVa7C4IISiMlXIdee2yjjZne2dufJh",
    'EDSR': "https://drive.google.com/uc?export=download&id=13E-EKIdHW6QyrZiLE8WvvDcJ1vnP9RgS",
}

# -----------------------------------------
# --- Perform SRR with fine-tuned models ---
# -----------------------------------------

def main(args=None):
    
    # Get inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('--LR-input', type=str, required=True)
    parser.add_argument('--model-name', type=str, choices=["EDSR", "RealESRGAN"], required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--no-flip-axes', action='store_true', default=False)
    
    args = parser.parse_args(args if args is not None else sys.argv[1:])
    model_name = args.model_name
    
    print('-------------------------------------------------')
    
    # Check inputs
    if not os.path.isfile(args.LR_input):
        raise ValueError('Input image does not exist.')
        
    if not os.path.isdir(os.path.dirname(args.output)):
        raise ValueError('Output directory does not exist.')
    
    # Check existence of weights directory
    weights_dir = os.path.join(repo_dir, "weights")
    if not os.path.isdir(weights_dir):
        raise RuntimeError(f'Weights directory not found in {weights_dir}')
    
    # Check existence of fine-tuned weights, try to download it 
    weights_fpath = os.path.join(weights_dir, model_name + "_finetuned.pth")
    if not os.path.isfile(weights_fpath):
        #raise ValueError('File with weights not found in', weights_fpath)
        print(f'Fine-tuned weights not found in {weights_fpath}, trying to download it...')
        import gdown
        gdown.download(FT_WEIGHTS_URLS[model_name], weights_fpath)

    else: 
        print(f'Fine-tuned weights found in {weights_fpath}')
        
    # Detect device/GPU
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    
    print('-------------------------------------------------')
        
    # Read input image
    LR = nib.load(args.LR_input)
    
    print("LR image:", args.LR_input)
    print('Image array shape:', LR.header['dim'][1:4])
    print('Voxel size:', LR.header['pixdim'][1:4])
    print('Orientation of voxel axes:', nib.aff2axcodes(LR.affine))
    
    original_ori = nib.io_orientation(LR.affine)
    flip_axes = (np.min(original_ori[:,1]) < 0) and not args.no_flip_axes
    
    if flip_axes:
        print("Flipping axes of LR input")
        reorient = original_ori
        reorient[:,0] = [0,1,2]
        print(reorient)
        LR = LR.as_reoriented(reorient)
        print('New orientation of voxel axes:', nib.aff2axcodes(LR.affine))
        
    print('-------------------------------------------------')
    print(f'Loading fine-tuned {model_name} model')
    
    # Get model
    if model_name == "EDSR":
        from prettier.models.edsr import EDSR
        model = EDSR(n_colorchannels = 3, scale = 4)
        model.load_state_dict(torch.load(weights_fpath)['model_weights'])
        scale_factor = 255
        
    elif model_name == "RealESRGAN":
        from basicsr.archs.rrdbnet_arch import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model.load_state_dict(torch.load(weights_fpath)['generator_weights'])
        scale_factor = None
        
    else:
        print("model name not valid")
    
    model.eval()
    model = model.to(device)
    
    # Reconstruct volume
    rec, v2w = reconstruct_volume(
        LR, 
        model, 
        device,
        batch_size = args.batch_size,
        scale_factor = scale_factor,
        slices_as_channels = True,
        select_middle = False,
        return_vol_list = False,
        combine_vol_method = COMBINE_VOL_METHOD,
    )
    
    hr_nib = nib.Nifti1Image(rec, v2w)
    
    # Flip back axes
    if flip_axes:
        print('Orientation of voxel axes:', nib.aff2axcodes(hr_nib.affine))
        print("Flipping axes of HR output")
        hr_nib = hr_nib.as_reoriented(reorient)
        print('New orientation of voxel axes:', nib.aff2axcodes(hr_nib.affine))
            
    # Save nifti image
    nib.save(hr_nib, args.output) 
    
    print("Reconstructed image saved in", args.output)

if __name__ == '__main__':
    main()