import numpy as np
from scipy.ndimage import gaussian_filter

# Fourier Burst Accumulation
# See https://github.com/remicongee/Fourier-Burst-Accumulation
# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7120097&tag=1
def fba_2d_onechannel(img_list, p = 11, sigma = 5):
    
    F_img = [np.fft.rfft2(img) for img in img_list]
    W_img = [gaussian_filter(np.abs(fimg), sigma = sigma) ** p for fimg in F_img]
    sum_W = np.sum(W_img, axis = 0)
    U = np.sum([ fimg*(wimg/sum_W) for fimg, wimg in zip(F_img, W_img) ], axis = 0) 
    
    return np.fft.irfft2(U)

def fba_nd_onechannel(img_list, p = 11, sigma = 5):
    
    F_img = [np.fft.rfftn(img) for img in img_list]
    W_img = [gaussian_filter(np.abs(fimg), sigma = sigma) ** p for fimg in F_img]
    sum_W = np.sum(W_img, axis = 0)
    U = np.sum([ fimg*(wimg/sum_W) for fimg, wimg in zip(F_img, W_img) ], axis = 0) 
    
    return np.fft.irfftn(U)

# Pad each RGB channel 
def pad_rgb_channels(rgb_slices): 
    
    r_channel = np.pad(rgb_slices[:,0,:,:], ((0,2),(0,0),(0,0)))
    g_channel = np.pad(rgb_slices[:,1,:,:], ((1,1),(0,0),(0,0)))
    b_channel = np.pad(rgb_slices[:,2,:,:], ((2,0),(0,0),(0,0)))
    outarray = np.stack((r_channel, g_channel, b_channel), axis = 1)
    
    return outarray

# Select method to combine information in RGB channels
def combine_channels(slc, method = "average", 
                     weights = np.array([1, 1, 1]), 
                     fba_p = 1., fba_sigma = 1.):
    
    if (method == "fba"):
        return fba_2d_onechannel(slc, p = fba_p, sigma = fba_sigma)
    
    else:
        return np.average(slc, axis = 0, weights = weights)

# Select method to combine volumes
def combine_volumes(vol_list, method = "average", fba_p = 1., fba_sigma = 1.):
    
    if (method == "fba"):
        return fba_nd_onechannel(vol_list, p = fba_p, sigma = fba_sigma)
    
    else:
        return np.mean(vol_list, axis = 0)
    



