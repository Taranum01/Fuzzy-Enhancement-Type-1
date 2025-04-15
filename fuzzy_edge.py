import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PATH = Path("/Users/taranumwasu/Downloads/fuzzy-image-contrast-enhancement-main/images/")
data = [str(p) for p in PATH.glob('*')]

# Gaussian Function:
def Gaussian(x, mean, std):
    return np.exp(-0.5 * np.square((x - mean) / std))

# Membership Functions for Edges:
def VeryWeakEdge(x, M):
    return Gaussian(x, 0, M / 6)

def WeakEdge(x, M):
    return Gaussian(x, M / 4, M / 6)

def MediumEdge(x, M):
    return Gaussian(x, M / 2, M / 6)

def StrongEdge(x, M):
    return Gaussian(x, 3 * M / 4, M / 6)

def VeryStrongEdge(x, M):
    return Gaussian(x, M, M / 6)

plt.figure(figsize=(20,5))
i = 1
for M in (128, 64, 192):
    x = np.arange(-50, 306)
    
    ED = VeryWeakEdge(x, M)
    VD = WeakEdge(x, M)
    Da = MediumEdge(x, M)
    SD = StrongEdge(x, M)
    SB = VeryStrongEdge(x, M)


    plt.subplot(3, 1, i)
    i += 1
    plt.plot(x, ED, 'k-.',label='VeryWeakEdge', linewidth=1)
    plt.plot(x, VD, 'k-',label='WeakEdge', linewidth=2)
    plt.plot(x, Da, 'g-',label='MediumEdge', linewidth=2)
    plt.plot(x, SD, 'b-',label='StrongEdge', linewidth=2)
    plt.plot(x, SB, 'r-',label='VeryStrongEdge', linewidth=2)
    plt.plot((M, M), (0, 1), 'm--', label='M', linewidth=2)
    plt.plot((0, 0), (0, 1), 'k--', label='MinIntensity', linewidth=2)
    plt.plot((255, 255), (0, 1), 'k--', label='MaxIntensity', linewidth=2)
    plt.xlim(-50, 305)
    plt.ylim(0.0, 1.01)
    plt.title(f'M={M}')
plt.legend()
plt.xlabel('Pixel intensity')
plt.ylabel('Degree of membership')
plt.show()

# Fuzzy Inference:
def FuzzyEdgeInference(x, M):
    VW = VeryWeakEdge(x, M)
    W = WeakEdge(x, M)
    M = MediumEdge(x, M)
    S = StrongEdge(x, M)
    VS = VeryStrongEdge(x, M)
    
    return np.maximum.reduce([VW, W, M, S, VS])

def ApplyFuzzyEdgeEnhancement(image):
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    
    # Calculate gradient magnitude
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Calculate M value (mean gradient magnitude)
    M = np.mean(grad_mag)
    
    # Apply fuzzy edge enhancement
    enhanced_edges = FuzzyEdgeInference(grad_mag, M)
    
    # Normalize enhanced edges to the range [0, 255]
    enhanced_edges = cv2.normalize(enhanced_edges, None, 0, 255, cv2.NORM_MINMAX)
    enhanced_edges = enhanced_edges.astype(np.uint8)
    
    return enhanced_edges

image = cv2.imread(data[0])

# Apply Fuzzy Edge Enhancement
enhanced_image = ApplyFuzzyEdgeEnhancement(image)

# Display the result
# cv2.imshow('Original Image', image)
# cv2.imshow('Enhanced Image', enhanced_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
