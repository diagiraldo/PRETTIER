# PRETTIER

<img src="figures/example_simulatedLR.png?raw=True" width="800px" style="margin:0px 0px"/>

<p> This repository contains ...</p>



## Requirements

This code depends on:
- [NiBabel](https://nipy.org/nibabel/) 
- [PyTorch](https://pytorch.org/), this tool has been developed and tested with [PyTorch 2.1.1](https://pytorch.org/get-started/previous-versions/#v211) and CUDA 11.8
- [BasicSR](https://github.com/XPixelGroup/BasicSR) is needed to import the generator architecture ([RRDBNet](https://basicsr.readthedocs.io/en/latest/api/basicsr.archs.rrdbnet_arch.html#basicsr.archs.rrdbnet_arch.RRDBNet)) of the RealESRGAN.

See full list in [`requirements.txt`](requirements.txt)

## Fine-tuned Models

| Model | Info | Fine-tuned weights |
| --- | ----------- | ---|
| RealESRGAN | [Paper](https://arxiv.org/abs/2107.10833), [Repository](https://github.com/xinntao/Real-ESRGAN) | [`RealESRGAN_finetuned.pth`](https://drive.google.com/file/d/15xWVa7C4IISiMlXIdee2yjjZne2dufJh/view?usp=drive_link) |
| EDSR | [Paper](https://arxiv.org/abs/1707.02921), [Repository](https://github.com/sanghyun-son/EDSR-PyTorch/) | [`EDSR_finetuned.pth`](https://drive.google.com/file/d/13E-EKIdHW6QyrZiLE8WvvDcJ1vnP9RgS/view?usp=drive_link) |

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

## Contact

Diana L. Giraldo Franco [@diagiraldo](https://github.com/diagiraldo)

