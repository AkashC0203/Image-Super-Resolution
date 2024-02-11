# Image Super-Resolution Using FSRCNN and SRGAN

## Overview
This repository showcases implementations of two advanced deep learning models for image super-resolution: Fast Super-Resolution Convolutional Neural Network (FSRCNN) and Super-Resolution Generative Adversarial Network (SRGAN). Aimed at upscaling low-resolution images with high fidelity to the original, these models leverage different techniques for enhancing image resolution. The repository includes `FSRCNN.py` and `SRGAN.ipynb` as the primary implementation files.

## Models Description

### FSRCNN (FSRCNN.py)
The FSRCNN model is encapsulated in `FSRCNN.py`. It acts as a baseline for performance comparison, utilizing a series of convolutional layers to improve low-resolution images. Its architecture comprises feature extraction, shrinking, non-linear mapping, expanding, and deconvolution layers. Optimized for the DIV2K dataset, this model has been adjusted to suit the specific input tensor requirements of the dataset.

### SRGAN (SRGAN.ipynb)
The `SRGAN.ipynb` file implements the SRGAN model. This model is based on a generative adversarial network framework and is designed to upscale low-resolution images into high-resolution ones. The generator in SRGAN focuses on upscaling the image, while the discriminator differentiates between the generated high-resolution images and real high-resolution images. This adversarial process, combined with a specialized loss function that includes content and adversarial loss, enables SRGAN to produce images that are not only high in resolution but also excel in perceptual quality.

## Dataset
Both models are trained and tested using the DIV2K dataset, which contains a variety of 2K resolution images. This dataset offers both low-resolution (downsampled) and high-resolution versions of images, making it an ideal resource for training and testing super-resolution models.

## Results and Evaluation
The models are evaluated using the PSNR metric, which measures the quality of reconstructed images. In our experiments, SRGAN achieved a significantly higher average PSNR value of 28.064 dB, compared to FSRCNN's 15.9588 dB. This indicates the superior capability of SRGAN in generating high-resolution images that are closer in quality to the original images, demonstrating its effectiveness in both upscaling and enhancing image resolution.
