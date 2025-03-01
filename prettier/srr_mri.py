# Diana Giraldo
# Aug 2023
# Last update: Dec 2024

from time import time
import numpy as np
import cv2

from .utils.reconstruction import combine_volumes, pad_rgb_channels, combine_channels
from .utils.image_slices import ImageSlices
from .utils.affine_transform import adjust_affine_transform
from .apply_models import apply_model_dataset

DEFAULT_NUM_WORKERS = 2
SLC_COMB_WEIGHTS = np.array([0.25, 0.5, 0.25])

# Function to upsample slices of a volume
def upsample_slices(
    in_nib_img, 
    slicing_dim, 
    out_shape,
    upsample_model,
    device,
    batch_size = 10,
    scale_factor = None,
    slices_as_channels = True,
    select_middle_channel = True,
    combine_channels_weights = SLC_COMB_WEIGHTS,
    interpolate_lr = False,
    print_info = True,
    num_workers = DEFAULT_NUM_WORKERS,
):
    
    # Create Dataset with LR slices
    lr_slices = ImageSlices(
        in_nib_object = in_nib_img,
        slice_dim = slicing_dim,
        preproces_training = True,
        scale_intensity = True,
        slices_as_channels = slices_as_channels
    )
    
    # Desired slice shape
    hr_slc_shape = np.delete(np.array(out_shape), slicing_dim).astype(int)
    # Fix to an external bug I don't want to spend time on
    hr_slc_shape = tuple([int(hr_slc_shape[0]), int(hr_slc_shape[1])])
    
    # Apply model to slices
    t0 = time()
    out = apply_model_dataset(
        upsample_model,
        lr_slices,
        device,
        batch_size = batch_size,
        scale_factor = scale_factor,
        interpolate_input = interpolate_lr,
        resize_model_output = not interpolate_lr,
        output_shape = hr_slc_shape,
        show_progress = print_info,
        num_workers = num_workers,
    )
    t1 = time()
    if print_info: print("Inference time:", t1 - t0)
    
    outarray = np.concatenate(out, axis = 0)
    
    # Adjust channels
    if slices_as_channels:
        outarray = pad_rgb_channels(outarray)
        
    # Obtain one greyscale image per slice
    if not slices_as_channels:
            out = [ cv2.cvtColor(np.transpose(slc, (1,2,0)), cv2.COLOR_BGR2GRAY) for slc in outarray ]
    else:
        if select_middle_channel:
            out = [ slc[1,:,:].squeeze() for slc in outarray ]
        else:
            out = [ combine_channels(slc, method = "average", weights = combine_channels_weights) for slc in outarray ]

    # Stack slices and recover intensity range
    recvol = np.stack(out, axis = slicing_dim) * lr_slices.IMG_scaleint_factor
    
    return recvol

#-------------------------------------------------------------------

def reconstruct_volume(
    LR_nib_img,
    model,
    device,
    batch_size = 10,
    scale_factor = None,
    slices_as_channels = True,
    select_middle_channel = False,
    interpolate_lr = False,
    return_vol_list = False,
    combine_vol_method = "average",
    fba_p = None, fba_sigma = None,
    print_info = True,
    num_workers = 2,
):
    
    LR_voxelsize = np.array(LR_nib_img.header.get_zooms())
    scaling_check = LR_voxelsize[2]/LR_voxelsize[0]
    if scaling_check.is_integer():
        LR_scaling = np.array([1, 1, scaling_check])
    else:
        #LR_scaling = np.round(LR_voxelsize)
        LR_scaling = np.round(np.array([1, 1, LR_voxelsize[2]]))
    HR_shape = (np.array(LR_nib_img.shape)*LR_scaling).astype(int)
    if print_info:
        print("-------------------------------------------")
        print("Scaling factor:", LR_scaling)
        print("HR image array shape:", HR_shape)
    slicing_dims = np.delete(np.arange(3), LR_scaling.argmax()).tolist()
    
    # Get HR transform
    LR_v2w = LR_nib_img.affine.astype(np.float64)
    HR_v2w = adjust_affine_transform(LR_v2w, LR_scaling)

    rec_list = []
    t0 = time()
    
    # Upsample slices in differnt slicing dimensions
    for slc_dim in slicing_dims:
    
        if print_info:
            print("-------------------------------------------")
            print("Slicing dimension:", slc_dim)

        recvol = upsample_slices(
            LR_nib_img, slc_dim, 
            HR_shape, 
            model, 
            device,
            batch_size = batch_size,
            scale_factor = scale_factor,
            slices_as_channels = slices_as_channels,
            select_middle_channel = select_middle_channel,
            interpolate_lr = interpolate_lr,
            print_info = print_info,
            num_workers = num_workers,
        )

        rec_list.append(recvol)
        
          
    # Combine volumes
    if print_info:
        print("-------------------------------------------")
        print("Combining volumes")
        
    HR_data = combine_volumes(
        rec_list, 
        method = combine_vol_method, 
        fba_p = fba_p, 
        fba_sigma = fba_sigma,
    )  
    
    t1 = time()
    if print_info: print("Total reconstruction time:", t1 - t0)

    if return_vol_list:
        return HR_data, HR_v2w, rec_list
    
    else: 
        return HR_data, HR_v2w