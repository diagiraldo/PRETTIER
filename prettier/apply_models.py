# Diana Giraldo
# May, 2023

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
import cv2
from tqdm import tqdm

def apply_model_dataset(
    model,
    data_set,
    device,
    batch_size = 5,
    scale_factor = None,
    resize_model_output = False, 
    output_shape = None,
    show_progress = True,
    num_workers = 2,
):
    model = model.to(device)
    
    # Make loader
    data_loader = DataLoader(
        dataset = data_set, 
        batch_size = batch_size, 
        shuffle = False,
        num_workers = num_workers, 
        pin_memory = torch.cuda.is_available()
    )
    
    out = []
    with torch.no_grad():
        with tqdm(data_loader, unit="batch", disable=(not show_progress)) as teval:
            for data in teval:
                tmp_input = data.to(device)
                if scale_factor : tmp_input *= scale_factor
                model_output = model(tmp_input)
                if scale_factor : model_output /= scale_factor
                if resize_model_output:
                    model_output = resize(model_output, size = output_shape, antialias=True)
                model_output = torch.clamp(model_output, min=0., max=1.).data.float().cpu().numpy()
                out.append(model_output)

    return out

