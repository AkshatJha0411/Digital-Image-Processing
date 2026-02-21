import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

def display_images(images, titles, cmap='gray'):
    """Helper function to display multiple images in a row."""
    plt.figure(figsize=(20, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        if len(images[i].shape) == 3:
            # Convert BGR to RGB for matplotlib
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Load a standard image (ensure 'input_image.jpg' is in your directory)
# Using '0' for grayscale load where appropriate
img_path = 'input.jpg' 
original_gray = cv2.imread(img_path, 0)
original_color = cv2.imread(img_path)

if original_gray is None:
    print("Error: Image not found. Please add an image named 'input.jpg' to the directory.")
else:
    # ==========================================
    # AIM 1: Image Sampling and Quantization
    # ==========================================
    print("--- Executing Aim 1: Sampling & Quantization ---")
    
    # Sampling (Spatial Resolution reduction)
    # Reducing resolution by taking every k-th pixel
    k = 4
    sampled_img = original_gray[::k, ::k]
    
    # Quantization (Intensity Level reduction)
    # Reducing from 8-bit (256 levels) to 3-bit (8 levels)
    # Formula: floor(img / (256/levels)) * (256/levels)
    levels = 8
    ratio = 256 / levels
    quantized_img = np.floor(original_gray / ratio) * ratio
    quantized_img = quantized_img.astype(np.uint8)

    display_images([original_gray, sampled_img, quantized_img], 
                   ['Original', f'Sampled (1/{k})', f'Quantized ({levels} levels)'])

    # ==========================================
    # AIM 2: Analysis of Intensity, Resolution & Histogram
    # ==========================================
    print("--- Executing Aim 2: Intensity & Histogram Analysis ---")
    
    # 1. Spatial Resolution (2x and 4x reduction via resizing)
    h, w = original_gray.shape
    res_2x = cv2.resize(original_gray, (w//2, h//2))
    res_4x = cv2.resize(original_gray, (w//4, h//4))
    
    display_images([original_gray, res_2x, res_4x], 
                   ['Original', 'Resolution / 2', 'Resolution / 4'])

    # 2. Histogram of Color Channels
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(10, 5))
    plt.title('Histogram of Color Channels')
    for i, color in enumerate(colors):
        # Calculate histogram for each channel
        hist = cv2.calcHist([original_color], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

    # ==========================================
    # AIM 3: Intensity Transformations
    # ==========================================
    print("--- Executing Aim 3: Intensity Transformations ---")
    
    # 1. Negative Transformation: s = L - 1 - r
    negative_img = 255 - original_gray
    
    # 2. Log Transformation: s = c * log(1 + r)
    c = 255 / np.log(1 + np.max(original_gray))
    log_img = c * (np.log(original_gray + 1))
    log_img = np.array(log_img, dtype=np.uint8)
    
    # 3. Gamma (Power-Law) Transformation: s = c * r^gamma
    gamma = 2.0  # Gamma > 1 darkens, Gamma < 1 brightens
    gamma_img = np.array(255 * (original_gray / 255) ** gamma, dtype='uint8')
    
    # 4. Histogram Equalization
    hist_eq_img = cv2.equalizeHist(original_gray)
    
    display_images([original_gray, negative_img, log_img, gamma_img, hist_eq_img],
                   ['Original', 'Negative', 'Log Trans', f'Gamma ({gamma})', 'Hist Equalization'])

    # ==========================================
    # AIM 4: Image Reconstruction
    # ==========================================
    print("--- Executing Aim 4: Image Reconstruction ---")
    
    # 1. Image Inpainting (Restoring missing parts)
    # Create a damaged image with a black square
    damaged_img = original_color.copy()
    mask = np.zeros(original_color.shape[:2], dtype=np.uint8)
    # Create a corruption (black square)
    cv2.rectangle(damaged_img, (50, 50), (100, 100), (0, 0, 0), -1)
    cv2.rectangle(mask, (50, 50), (100, 100), 255, -1)
    
    # Inpaint using Telea method
    inpainted_img = cv2.inpaint(damaged_img, mask, 3, cv2.INPAINT_TELEA)
    
    # 2. Fourier Transform Reconstruction
    dft = np.fft.fft2(original_gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift)) # For visualization
    
    # Reconstruct (Inverse FFT)
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # 3. Inverse Filtering (Simulated using Wiener Filter for deblurring/denoising)
    # Note: True inverse filtering requires knowing the degradation function.
    # We use scipy's wiener filter as an approximation for noise removal.
    wiener_img = wiener(original_gray, (5, 5))
    
    display_images([damaged_img, inpainted_img, wiener_img, img_back], 
                   ['Damaged', 'Inpainted', 'Wiener (Deblur/Denoise)', 'Fourier Reconstruction'])

    # ==========================================
    # AIM 5: Interpolation (Decimation & Upscaling)
    # ==========================================
    print("--- Executing Aim 5: Interpolation Analysis ---")
    
    # Decimation (Downsampling by factor of 4)
    small_img = cv2.resize(original_color, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    
    # Upscaling using different interpolation methods
    nearest = cv2.resize(small_img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    bilinear = cv2.resize(small_img, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    bicubic = cv2.resize(small_img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    display_images([small_img, nearest, bilinear, bicubic], 
                   ['Decimated (Low Res)', 'Nearest Neighbor', 'Bilinear', 'Bicubic'])