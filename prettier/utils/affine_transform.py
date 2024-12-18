import numpy as np

# Function to adjust affine transform given a scaling of voxel sizes
def adjust_affine_transform(affine, scale_factor):
    new_R = np.dot(affine[:3, :3], np.linalg.inv(np.diag(scale_factor).astype(np.float64)))
    new_b = affine[:3, 3] - np.dot(new_R,((scale_factor - 1) / 2.))
    new_affine = np.block([[new_R, new_b.reshape(-1,1)],
                           [np.zeros((1, 3)), 1.]]).astype(np.float64)
    return new_affine