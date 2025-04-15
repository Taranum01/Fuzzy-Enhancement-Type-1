import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Markdown
from glob2 import glob
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy

from edge_enh import DefaultEdgeEnh
from fuzzy_edge import ApplyFuzzyEdgeEnhancement

PATH = Path("/Users/taranumwasu/Downloads/fuzzy-image-contrast-enhancement-main/images/")

# Gaussian Function:
def Gaussian(x, mean, std):
    return np.exp(-0.5*np.square((x-mean)/std))

# Membership Functions:
def ExtremelyDark(x, M):
    return Gaussian(x, -50, M/6)

def VeryDark(x, M):
    return Gaussian(x, 0, M/6)

def Dark(x, M):
    return Gaussian(x, M/2, M/6)

def SlightlyDark(x, M):
    return Gaussian(x, 5*M/6, M/6)

def SlightlyBright(x, M):
    return Gaussian(x, M+(255-M)/6, (255-M)/6)

def Bright(x, M):
    return Gaussian(x, M+(255-M)/2, (255-M)/6)

def VeryBright(x, M):
    return Gaussian(x, 255, (255-M)/6)

def ExtremelyBright(x, M):
    return Gaussian(x, 305, (255-M)/6)

plt.figure(figsize=(20,5))
i = 1
for M in (128, 64, 192):
    x = np.arange(-50, 306)
    
    ED = ExtremelyDark(x, M)
    VD = VeryDark(x, M)
    Da = Dark(x, M)
    SD = SlightlyDark(x, M)
    SB = SlightlyBright(x, M)
    Br = Bright(x, M)
    VB = VeryBright(x, M)
    EB = ExtremelyBright(x, M)


    plt.subplot(3, 1, i)
    i += 1
    plt.plot(x, ED, 'k-.',label='ExtremelyDark', linewidth=1)
    plt.plot(x, VD, 'k-',label='VeryDark', linewidth=2)
    plt.plot(x, Da, 'g-',label='Dark', linewidth=2)
    plt.plot(x, SD, 'b-',label='SlightlyDark', linewidth=2)
    plt.plot(x, SB, 'r-',label='SlightlyBright', linewidth=2)
    plt.plot(x, Br, 'c-',label='Bright', linewidth=2)
    plt.plot(x, VB, 'y-',label='VeryBright', linewidth=2)
    plt.plot(x, EB, 'y-.',label='ExtremelyBright', linewidth=1)
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

def OutputFuzzySet(x, f, M, thres):
    x = np.array(x)
    result = f(x, M)
    result[result > thres] = thres
    return result

def AggregateFuzzySets(fuzzy_sets):
    return np.max(np.stack(fuzzy_sets), axis=0)

def Infer(i, M, get_fuzzy_set=False):
    # Calculate degree of membership for each class
    VD = VeryDark(i, M)
    Da = Dark(i, M)
    SD = SlightlyDark(i, M)
    SB = SlightlyBright(i, M)
    Br = Bright(i, M)
    VB = VeryBright(i, M)
    
    # Fuzzy Inference:
    x = np.arange(-50, 306)
    Inferences = (
        OutputFuzzySet(x, ExtremelyDark, M, VD),
        OutputFuzzySet(x, VeryDark, M, Da),
        OutputFuzzySet(x, Dark, M, SD),
        OutputFuzzySet(x, Bright, M, SB),
        OutputFuzzySet(x, VeryBright, M, Br),
        OutputFuzzySet(x, ExtremelyBright, M, VB)
    )
    
    # Calculate AggregatedFuzzySet:
    fuzzy_output = AggregateFuzzySets(Inferences)
    
    # Calculate crisp value of centroid
    if get_fuzzy_set:
        return np.average(x, weights=fuzzy_output), fuzzy_output
    return np.average(x, weights=fuzzy_output)

plt.figure(figsize=(20,5))
i = 1
for pixel in (64, 96, 160, 192):
    M = 128
    x = np.arange(-50, 306)
    centroid, output_fuzzy_set = Infer(np.array([pixel]), M, get_fuzzy_set=True)
    plt.subplot(4, 1, i)
    i += 1
    # plt.plot(x, output_fuzzy_set, 'k-',label='FuzzySet', linewidth=2)
    # plt.plot((M, M), (0, 1), 'm--', label='M', linewidth=2)
    # plt.plot((pixel, pixel), (0, 1), 'g--', label='Input', linewidth=2)
    # plt.plot((centroid, centroid), (0, 1), 'r--', label='Output', linewidth=2)
    # plt.fill_between(x, np.zeros(356), output_fuzzy_set, color=(.9, .9, .9, .9))
    # plt.xlim(-50, 305)
    # plt.ylim(0.0, 1.01)
    # plt.xlabel('Output pixel intensity')
    # plt.ylabel('Degree of membership')
    # plt.title(f'input_pixel_intensity = {pixel}\nM = {M}')
# plt.legend()
# plt.show()

means = (64, 96, 128, 160, 192)
plt.figure(figsize=(25,5))
for i in range(len(means)):
    M = means[i]
    x = np.arange(256)
    y = np.array([Infer(np.array([i]), M) for i in x])
    # plt.subplot(1, len(means), i+1)
    # plt.plot(x, y, 'r-', label='IO mapping')
    # plt.xlim(0, 256)
    # plt.ylim(-50, 355)
    # plt.xlabel('Input Intensity')
    # plt.ylabel('Output Intensity')
    # plt.title(f'M = {M}')
# plt.show()


# Proposed fuzzy method

# def FuzzyContrastEnhance(rgb, color_space='LAB'):
#     color_space_dict = {
#         'LAB': (cv2.COLOR_RGB2LAB, cv2.COLOR_LAB2RGB, 0),
#         'XYZ': (cv2.COLOR_RGB2XYZ, cv2.COLOR_XYZ2RGB, 0),
#         'YCbCr': (cv2.COLOR_RGB2YCrCb, cv2.COLOR_YCrCb2RGB, 0),
#         'HSV': (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB, 0),
#         'HLS': (cv2.COLOR_RGB2HLS, cv2.COLOR_HLS2RGB, 1),
#         'LUV': (cv2.COLOR_RGB2LUV, cv2.COLOR_LUV2RGB, 0),
#         'YUV': (cv2.COLOR_RGB2YUV, cv2.COLOR_YUV2RGB, 0)
#     }

#     if color_space not in color_space_dict:
#         raise ValueError(f"Unsupported color space: {color_space}")

#     # Get conversion codes and L channel index
#     convert_to, convert_from, l_channel = color_space_dict[color_space]

#     # Convert RGB to the selected color space
#     lab = cv2.cvtColor(rgb, convert_to)
#     l = lab[:, :, l_channel]

#     # Calculate M value
#     M = np.mean(l)
#     if M < 128:
#         M = 127 - (127 - M) / 2
#     else:
#         M = 128 + M / 2

#     # Precompute the fuzzy transform
#     x = list(range(-50, 306))
#     FuzzyTransform = dict(zip(x, [Infer(np.array([i]), M) for i in x]))

#     # Apply the transform to L channel
#     u, inv = np.unique(l, return_inverse=True)
#     l = np.array([FuzzyTransform[i] for i in u])[inv].reshape(l.shape)

#     # Min-max scale the output L channel to fit (0, 255)
#     Min = np.min(l)
#     Max = np.max(l)
#     lab[:, :, l_channel] = (l - Min) / (Max - Min) * 255

#     # Convert back to RGB from the selected color space
#     return cv2.cvtColor(lab, convert_from)
    

def FuzzyContrastEnhance(rgb):
    # Convert RGB to LAB
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    
    # Get L channel
    l = lab[:, :, 0]
    
    # Calculate M value
    M = np.mean(l)
    if M < 128:
        M = 127 - (127 - M)/2
    else:
        M = 128 + M/2
        
    # Precompute the fuzzy transform
    x = list(range(-50,306))
    FuzzyTransform = dict(zip(x,[Infer(np.array([i]), M) for i in x]))
    
    # Apply the transform to l channel
    u, inv = np.unique(l, return_inverse = True)
    l = np.array([FuzzyTransform[i] for i in u])[inv].reshape(l.shape)
    
    # Min-max scale the output L channel to fit (0, 255):
    Min = np.min(l)
    Max = np.max(l)
    lab[:, :, 0] = (l - Min)/(Max - Min) * 255
    
    # Convert LAB to RGB
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# Traditional method of histogram equalization
def HE(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# Contrast Limited Adaptive Histogram Equalization
def CLAHE(rgb):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# Define convolution function
def apply_convolution(image, kernel, stride=1, padding='valid'):
    if padding == 'same':
        pad_h = (kernel.shape[0] - 1) // 2
        pad_w = (kernel.shape[1] - 1) // 2
        image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    output_height = (image.shape[0] - kernel.shape[0]) // stride + 1
    output_width = (image.shape[1] - kernel.shape[1]) // stride + 1
    output = np.zeros((output_height, output_width))

    for y in range(0, image.shape[0] - kernel.shape[0] + 1, stride):
        for x in range(0, image.shape[1] - kernel.shape[1] + 1, stride):
            output[y // stride, x // stride] = np.sum(image[y:y + kernel.shape[0], x:x + kernel.shape[1]] * kernel)

    return output

# Edge Filters
vertical_edge_detection = np.array([[ 1,  0, -1],
                                    [ 1,  0, -1],
                                    [ 1,  0, -1]])

horizontal_edge_detection = np.array([[ 1,  1,  1],
                                      [ 0,  0,  0],
                                      [-1, -1, -1]])

edge_enhancement = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])


def evaluate_methods(original_rgb):
    methods = {
        "Original": original_rgb,
        "Fuzzy": FuzzyContrastEnhance(original_rgb),
        "HE": HE(original_rgb),
        "CLAHE": CLAHE(original_rgb)
    }

    print(f"{'Method':<10} | {'PSNR':<8} | {'SSIM':<8} | {'Entropy':<8}")
    print("-" * 40)

    for name, img in methods.items():
        if name != "Original":
            psnr_val = psnr(original_rgb, img)
            ssim_val = ssim(original_rgb, img, channel_axis=2)
        else:
            psnr_val = ssim_val = float('nan')

        entropy_val = shannon_entropy(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

        print(f"{name:<10} | {psnr_val:<8.4f} | {ssim_val:<8.4f} | {entropy_val:<8.4f}")

        plt.figure(figsize=(4,4))
        plt.imshow(img)
        plt.title(name)
        plt.axis('off')
        plt.show()

# Load images in RGB format
data = np.array([cv2.cvtColor(cv2.imread(str(p)), cv2.IMREAD_COLOR) for p in PATH.glob('*')])

for i in range(data.shape[0]):
    rgb_img = data[i]
    evaluate_methods(rgb_img)  # Evaluate performance
    # rgb_img = cv2.imread(data[0], cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    # lab_img = FuzzyContrastEnhance(rgb_img, 'LAB')
    # lab_gray = cv2.cvtColor(lab_img, cv2.COLOR_RGB2GRAY)

    # enhanced_edges = cv2.filter2D(gray_img, -1, edge_enhancement)

    # vertical_edges = apply_convolution(gray_img, vertical_edge_detection, padding='same')
    # horizontal_edges = apply_convolution(gray_img, horizontal_edge_detection, padding='same')
    # enhanced_edges = apply_convolution(gray_img, edge_enhancement, padding='same')

    he = HE(rgb_img)
    clahe = CLAHE(rgb_img) 

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    org_img = plt.imshow(rgb_img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    enh_img = plt.imshow(FuzzyContrastEnhance(rgb_img))
    plt.title('LAB Fuzzy Contrast Enhance')

    # plt.subplot(2, 2, 3)
    # enh_img = plt.imshow()
    # plt.title('horizontal_edges')

    # plt.subplot(2, 2, 4)
    # enh_img = plt.imshow(FuzzyContrastEnhance(rgb_img, 'HSV'))
    # plt.title('HSV Fuzzy Contrast Enhance')

    plt.subplot(2, 2, 3)
    # plt.imshow(ApplyFuzzyEdgeEnhancement(rgb_img))
    plt.imshow(ApplyFuzzyEdgeEnhancement(rgb_img))
    plt.title('Fuzzy Edge Enhance')

    plt.subplot(2, 2, 4)
    plt.imshow(DefaultEdgeEnh(rgb_img))
    plt.title('Fuzzy Contrast-Edge Enhance')

    # plt.tight_layout()
    plt.show()
    