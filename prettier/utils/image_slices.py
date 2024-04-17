# Diana Giraldo
# Class to get image slices from MRI nifti file

import numpy as np
import torch
from torch.utils.data import Dataset

import nibabel as nib
import cv2
from ..utils.mrislicing import get_slice

class ImageSlices(Dataset):
    
    #Constructor
    def __init__(
        self,
        in_nib_object,
        slice_dim,
        preproces_training = False,
        scale_intensity = False,
        slices_as_channels = False,
        interpolate_slices = False,
        
    ):
        
        self.opt_preproces_training = preproces_training
        self.opt_scale_intensity = scale_intensity
        self.opt_slices_as_channels = slices_as_channels
        self.opt_interpolate_slices = interpolate_slices
        self.slicing_dim = slice_dim
        
        # Info nifti image
        self.IMG = in_nib_object
        self.IMG_shape = self.IMG.shape
        
        if scale_intensity:
            self.IMG_scaleint_factor = np.percentile(self.IMG.get_fdata(), 99.5)*1.25 
        else:
            self.IMG_scaleint_factor = 1.
            
    
    # Get item: one slice
    def __getitem__(self, index):
        
        x = get_slice(
            self.IMG, 
            index+1, 
            self.slicing_dim,
            slices_as_channels = self.opt_slices_as_channels
        ).astype(np.float32)

        if self.opt_scale_intensity:
            x = x/self.IMG_scaleint_factor
            
        
        if self.opt_preproces_training:
            if not self.opt_slices_as_channels:
                x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
            x = torch.from_numpy(np.transpose(x, (2, 0, 1))).float()
        
        return x     
        
    # Get length of dataset
    def __len__(self):
        n_slices = self.IMG_shape[self.slicing_dim]
        
        if not self.opt_slices_as_channels:
             return n_slices
        else:
            return (n_slices - 2)     


