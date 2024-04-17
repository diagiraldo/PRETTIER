# PRETTIER

This repository contains ...

## Summary

Description, add example figure
<img src="figures/example_simulatedLR_FLAIR.png?raw=True" width="800px" style="margin:0px 0px"/>

## Requirements

- Pytorch
- nibabel
- BasicSR 


## Fine-tuned Models

- RealESRGAN
- EDSR

## Usage
```
./prettier_mri_6mm_to_1mm.py --LR-input <lr_input> --model-name {EDSR,RealESRGAN} --output <output_image> [--gpu-id GPU_ID] [--batch-size BATCH_SIZE] [--no-flip-axes]
```

Example:
```
./prettier_mri_6mm_to_1mm.py --LR-input example/synth_LR_T1.nii.gz --model-name EDSR --output example/prettier_edsr_synth_T1.nii.gz
```

## Citations
Not yet.

