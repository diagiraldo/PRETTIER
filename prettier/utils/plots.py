import matplotlib.pyplot as plt
import numpy as np

# Function for visualization of orthogonal views 
def show_orthoslices(image, 
                     cross = None, 
                     figsize = (18,6), 
                     cmap = "gray", 
                     vmin = None, 
                     vmax = None, 
                     show_cross = False,
                     out_file = None):
    
    slices = [image[cross[0], :, :], image[:, cross[1], :], image[:, :, cross[2]]]
    dims = [("Dimension 2", "Dimension 3"), ("Dimension 1", "Dimension 3"), ("Dimension 1", "Dimension 2")]

    if vmin is None:
        vmin = np.min(image)
        
    if vmax is None:
        vmax = np.max(image)
        
    if show_cross:
        lines = [
            (cross[1], cross[2]),
            (cross[0], cross[2]),
            (cross[0], cross[1])
        ]
    
    # create plots and save image
    fig, axes = plt.subplots(1, len(slices), figsize = figsize)
    ims = []
    for i, slc in enumerate(slices):
        # subplots, labels and colorbars
        im = axes[i].imshow(slc.T, cmap = cmap, origin = "lower", interpolation = "none", vmin = vmin, vmax = vmax)
        axes[i].set_xlabel(dims[i][0])
        axes[i].set_ylabel(dims[i][1])
        axes[i].set_aspect(slc.shape[0]/slc.shape[1])
        axes[i].set_xticks([]) 
        axes[i].set_yticks([])
        
        if show_cross:
            axes[i].axvline(x = lines[i][0], c = "white", linestyle='--', lw = 0.5)
            axes[i].axhline(y = lines[i][1], c = "white", linestyle='--', lw = 0.5)
        
        ims.append(im)

    left = axes[0].get_position().x0
    bottom = axes[0].get_position().y1 + 0.05
    width = abs(axes[0].get_position().x0 - axes[2].get_position().x1)
    height = 0.02
    cax = fig.add_axes([left, bottom, width, height])
    fig.colorbar(ims[0], cax=cax, orientation="horizontal")
    # fig.set_facecolor("w")
    
    if out_file is not None:
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
    
    plt.show()
    plt.close()