import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Markdown
from glob2 import glob
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
import imageio  

def DefaultEdgeEnh(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    return edges

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

def FuzzyEdgeInference(x, M):
    VW = VeryWeakEdge(x, M)
    W = WeakEdge(x, M)
    M = MediumEdge(x, M)
    S = StrongEdge(x, M)
    VS = VeryStrongEdge(x, M)
    
    return np.maximum.reduce([VW, W, M, S, VS])

def ApplyFuzzyEdgeEnhancement(image):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    M = np.mean(grad_mag)
    enhanced_edges = FuzzyEdgeInference(grad_mag, M)
    enhanced_edges = cv2.normalize(enhanced_edges, None, 0, 255, cv2.NORM_MINMAX)
    enhanced_edges = enhanced_edges.astype(np.uint8)
    
    return enhanced_edges

def Gaussian(x, mean, std):
    epsilon = 1e-8  # Small value to avoid division by zero
    x_clipped = np.clip(x, -1000, 1000)  # Clip values to a reasonable range
    return np.exp(-0.5 * np.square((x_clipped - mean) / (std + epsilon)))

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

def plot_membership_functions():
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
    VD = VeryDark(i, M)
    Da = Dark(i, M)
    SD = SlightlyDark(i, M)
    SB = SlightlyBright(i, M)
    Br = Bright(i, M)
    VB = VeryBright(i, M)
    
    x = np.arange(-50, 306)
    Inferences = (
        OutputFuzzySet(x, ExtremelyDark, M, VD),
        OutputFuzzySet(x, VeryDark, M, Da),
        OutputFuzzySet(x, Dark, M, SD),
        OutputFuzzySet(x, Bright, M, SB),
        OutputFuzzySet(x, VeryBright, M, Br),
        OutputFuzzySet(x, ExtremelyBright, M, VB)
    )
    
    fuzzy_output = AggregateFuzzySets(Inferences)
    
    if get_fuzzy_set:
        return np.average(x, weights=fuzzy_output), fuzzy_output
    return np.average(x, weights=fuzzy_output)

def plot_inference_examples():
    plt.figure(figsize=(20,5))
    i = 1
    for pixel in (64, 96, 160, 192):
        M = 128
        x = np.arange(-50, 306)
        centroid, output_fuzzy_set = Infer(np.array([pixel]), M, get_fuzzy_set=True)
        plt.subplot(4, 1, i)
        plt.plot(x, output_fuzzy_set)
        plt.title(f'Input Pixel Intensity: {pixel}, Output: {centroid:.1f}')
        plt.xlim(-50, 305)
        plt.ylim(0, 1)
        i += 1
    plt.tight_layout()
    plt.show()

def FuzzyContrastEnhance(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0]
    
    M = np.mean(l)
    if M < 128:
        M = 127 - (127 - M)/2
    else:
        M = 128 + M/2
        
    x = list(range(-50,306))
    FuzzyTransform = dict(zip(x,[Infer(np.array([i]), M) for i in x]))
    
    u, inv = np.unique(l, return_inverse = True)
    l = np.array([FuzzyTransform[i] for i in u])[inv].reshape(l.shape)
    
    Min = np.min(l)
    Max = np.max(l)
    lab[:, :, 0] = (l - Min)/(Max - Min) * 255
    
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def HE(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def CLAHE(rgb):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def process_video(video_path, output_path=None, display_frames=False):
    """Process a video file frame by frame with fuzzy enhancement"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply fuzzy enhancement
        enhanced_frame = FuzzyContrastEnhance(frame_rgb)
        
        # Convert back to BGR for video writing/display
        enhanced_frame_bgr = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)
        
        if output_path:
            out.write(enhanced_frame_bgr)
            
        if display_frames:
            cv2.imshow('Original', frame)
            cv2.imshow('Enhanced', enhanced_frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        frame_count += 1
        print(f"Processed frame {frame_count}", end='\r')
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    print(f"\nFinished processing {frame_count} frames")

def process_gif(gif_path, output_path=None, display_frames=False):
    """Process a GIF file frame by frame with fuzzy enhancement"""
    gif = imageio.get_reader(gif_path)
    
    frames = []
    for frame in gif:

        if frame.ndim == 2:  
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:  
            frame_rgb = frame
            

        enhanced_frame = FuzzyContrastEnhance(frame_rgb)
        frames.append(enhanced_frame)
        
        if display_frames:
            plt.imshow(enhanced_frame)
            plt.axis('off')
            plt.show()
    
    if output_path:
        imageio.mimsave(output_path, frames, duration=gif.get_meta_data()['duration']/1000)
    
    return frames

def evaluate_methods(original_rgb, is_video_frame=False):
    methods = {
        "Original": original_rgb,
        "Fuzzy": FuzzyContrastEnhance(original_rgb),
        "HE": HE(original_rgb),
        "CLAHE": CLAHE(original_rgb)
    }

    if not is_video_frame:
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

    for name, img in methods.items():
        if not is_video_frame:
            plt.figure(figsize=(4,4))
            plt.imshow(img)
            plt.title(name)
            plt.axis('off')
            plt.show()

def process_image(image_path, output_path=None, display_results=True):
    """Process a single image with all enhancement methods"""
    rgb_img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    
    if display_results:
        evaluate_methods(rgb_img)
        
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(rgb_img)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        enhanced_img = FuzzyContrastEnhance(rgb_img)
        plt.imshow(enhanced_img)
        plt.title('LAB Fuzzy Contrast Enhance')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        fuzzy_edge = ApplyFuzzyEdgeEnhancement(rgb_img)
        plt.imshow(fuzzy_edge, cmap='gray')
        plt.title('Fuzzy Edge Enhance')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        default_edge = DefaultEdgeEnh(rgb_img)
        plt.imshow(default_edge, cmap='gray')
        plt.title('Default Edge Enhance')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))
    
    return {
        'original': rgb_img,
        'fuzzy_enhanced': enhanced_img,
        'fuzzy_edge': fuzzy_edge,
        'default_edge': default_edge
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process images, videos, or GIFs with fuzzy enhancement')
    parser.add_argument('input_path', type=str, help='Path to input file or directory')
    parser.add_argument('--output', type=str, help='Output path for processed file')
    parser.add_argument('--display', action='store_true', help='Display processed frames')
    args = parser.parse_args()

    input_path = Path(args.input_path)
    
    if input_path.is_dir():

        image_paths = [p for p in input_path.glob('*') 
                      if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        for img_path in image_paths:
            print(f"\nProcessing {img_path.name}")
            process_image(img_path, args.display)
            
    elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov']:

        process_video(str(input_path), args.output, args.display)
        
    elif input_path.suffix.lower() in ['.gif']:

        process_gif(str(input_path), args.output, args.display)
        
    elif input_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:

        process_image(input_path, args.display)
    else:
        print("Unsupported file format")

if __name__ == "__main__":
    plot_membership_functions()
    plot_inference_examples()
    
    main()
