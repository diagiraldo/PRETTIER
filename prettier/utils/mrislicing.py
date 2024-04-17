# Diana Giraldo
# Mayo, 2023

import numpy as np
import nibabel as nib

def get_slicingdims(lr_fpath, hr_fpath):
    LR = nib.load(lr_fpath)
    HR = nib.load(hr_fpath)
    LR2HR_ornt = nib.orientations.ornt_transform(start_ornt = nib.orientations.io_orientation(LR.affine), 
                                             end_ornt = nib.orientations.io_orientation(HR.affine))
    LR = LR.as_reoriented(LR2HR_ornt)
    LR_vox = np.round(np.array(LR.header.get_zooms()))
    slicing_dims = np.delete(np.arange(3), LR_vox.argmax()).tolist()
    return slicing_dims


def get_slice(im_nib, sliceidx, slice_dim, 
              slices_as_channels = False):
    
    if slice_dim == 0:
        slc = im_nib.dataobj[sliceidx, :, :] if not slices_as_channels else np.transpose(im_nib.dataobj[(sliceidx-1):(sliceidx+2), :, :], (1, 2, 0))
    elif slice_dim == 1:
        slc = im_nib.dataobj[:, sliceidx, :] if not slices_as_channels else np.transpose(im_nib.dataobj[:, (sliceidx-1):(sliceidx+2), :], (0, 2, 1))
    else:
        slc = im_nib.dataobj[:, :, sliceidx] if not slices_as_channels else im_nib.dataobj[:, :, (sliceidx-1):(sliceidx+2)]
        
    return slc

