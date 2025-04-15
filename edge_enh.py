import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# PATH = Path('/home/amirreza/proj/fuzzy/image-contrast-enhancement/images/')
# data = [str(p) for p in PATH.glob('*')]
# image = cv2.imread(data[0], cv2.IMREAD_COLOR)

def DefaultEdgeEnh(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

    return edges

    # Display original image and edges
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges Detected')
    plt.axis('off')

    plt.tight_layout()
    plt.show()