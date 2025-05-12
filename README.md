# Fuzzy Image, GIF, and Video Contrast Enhancement

This project implements a fuzzy logic-based contrast enhancement method that works seamlessly across **images**, **GIFs**, and **videos**. It leverages fuzzy sets and inference mechanisms to enhance visual clarity, especially in challenging conditions such as low contrast and high noise. The method is extensively tested on an **underwater infrared dataset**, which is known for its poor visibility and heavy noise artifacts.

## Features

- 🎯 **Fuzzy Contrast Enhancement**  
  Applies fuzzy logic to model pixel intensities, improving contrast and revealing hidden details in low-visibility content.

- 🖼️ **Multi-Format Support**  
  Supports static **images**, animated **GIFs**, and dynamic **videos** through a unified processing pipeline.

- ⚙️ **Traditional Enhancement Techniques for Comparison**  
  Includes:
  - Histogram Equalization (HE)
  - Contrast Limited Adaptive Histogram Equalization (CLAHE)

- 🧠 **Edge Detection & Enhancement**  
  Incorporates edge filters and fuzzy enhancement to sharpen object boundaries and improve structural detail.

- 📊 **Evaluation Metrics**  
  Compares results using:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - Shannon Entropy

- 🌊 **Special Focus: Underwater Infrared Dataset**  
  Optimized and evaluated on infrared images captured in underwater environments—ideal for testing in low-light, high-noise scenarios.

## Prerequisites

Make sure the following Python libraries are installed:

```bash
pip install opencv-python matplotlib numpy scikit-image ipython glob2 imageio
