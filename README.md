# Fuzzy-Image-Enhancement-Type-Ⅰ

This project implements a fuzzy logic-based image contrast enhancement method. It uses fuzzy sets and inference mechanisms to improve the contrast of images, with a focus on fuzzy edge enhancement and various traditional contrast enhancement methods for comparison.

## Features

- **Fuzzy Image Enhancement**: Enhances contrast by using fuzzy logic to model pixel intensities and improve image details, especially for low-contrast areas.
- **Traditional Methods**: Compares the fuzzy method with traditional image contrast techniques, such as:
  - Histogram Equalization (HE)
  - Contrast Limited Adaptive Histogram Equalization (CLAHE)
- **Edge Detection and Enhancement**: Applies edge filters to enhance image details and sharpness.
- **Evaluation**: Compares the performance of the enhancement methods using metrics like PSNR, SSIM, and Shannon Entropy.

## Prerequisites

Ensure you have the following dependencies installed:

- Python 3.x
- OpenCV (`cv2`)
- Matplotlib
- Numpy
- scikit-image
- IPython
- glob2
