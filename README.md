# PRETTIER

This repository contains the script to apply PRETTIER, a framework to perform super-resolution on structural MRI. This framework relies on convolutional neural networks (CNN) that have been fine-tune with data from T2-W FLAIR and T1-W MRIs of people with Multiple Sclerosis.  
We have developed and evaluated PRETTIER to increase the through-plane resolution of multi-slice MRI from 6mm to 1mm.

At this moment (May 2024), there are two fine-tuned CNN models to work within PRETTIER: EDSR and RealESRGAN.

<img src="figures/example_simulatedLR.png?raw=True" width="800px" style="margin:0px 0px"/>

The name **PRETTIER** comes from **P**erceptual super-**RE**solu**T**ion in mul**TI**ple scl**ER**osis. (You can say it is a rather convoluted explanation for a name, and I would agree. But you cannot say it isn't a pretty fitting name for a super-resolution method)

## Requirements

This code depends on:
- [NiBabel](https://nipy.org/nibabel/) 
- [PyTorch](https://pytorch.org/), this tool has been developed and tested with [PyTorch 2.1.1](https://pytorch.org/get-started/previous-versions/#v211) and CUDA 11.8
- [BasicSR](https://github.com/XPixelGroup/BasicSR) is needed to import the generator architecture ([RRDBNet](https://basicsr.readthedocs.io/en/latest/api/basicsr.archs.rrdbnet_arch.html#basicsr.archs.rrdbnet_arch.RRDBNet)) of RealESRGAN.

See full list in [`requirements.txt`](requirements.txt)

## Fine-tuned Models

| Model | Info | Fine-tuned weights |
| --- | ----------- | ---|
| RealESRGAN | [Paper](https://arxiv.org/abs/2107.10833), [Repository](https://github.com/xinntao/Real-ESRGAN) | [`RealESRGAN_finetuned.pth`](https://drive.google.com/file/d/15xWVa7C4IISiMlXIdee2yjjZne2dufJh/view?usp=drive_link) |
| EDSR | [Paper](https://arxiv.org/abs/1707.02921), [Repository](https://github.com/sanghyun-son/EDSR-PyTorch/) | [`EDSR_finetuned.pth`](https://drive.google.com/file/d/13E-EKIdHW6QyrZiLE8WvvDcJ1vnP9RgS/view?usp=drive_link) |

## Usage

```
./prettier_mri.py --LR-input <lr_input> --model-name {EDSR,RealESRGAN} --output <output_image> [--gpu-id GPU_ID] [--batch-size BATCH_SIZE] [--no-flip-axes]
```

Example:
```
./prettier_mri.py --LR-input demo_data/synth_LR_T1.nii.gz --model-name EDSR --output demo_data/prettier_edsr_synth_T1.nii.gz
```

## Citations

Not yet.

## Contact

Diana L. Giraldo Franco [@diagiraldo](https://github.com/diagiraldo)

