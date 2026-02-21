import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

img_path = 'input.jpg'
original_color = cv2.imread(img_path)
original_gray = cv2.imread(img_path, 0)

if original_color is None:
    print("Error: Image not found.")
else:
    # 1. Image Inpainting [cite: 26, 33]
    # Create a 'damaged' image with a mask
    damaged = original_color.copy()
    mask = np.zeros(original_color.shape[:2], dtype=np.uint8)
    
    # Draw a black rectangle (damage)
    cv2.rectangle(damaged, (50, 50), (120, 120), (0, 0, 0), -1)
    cv2.rectangle(mask, (50, 50), (120, 120), 255, -1)
    
    # Reconstruct using Telea algorithm
    inpainted = cv2.inpaint(damaged, mask, 3, cv2.INPAINT_TELEA)

    # 2. Fourier Reconstruction [cite: 28, 35]
    # Analysis of frequencies and inverse transform
    dft = np.fft.fft2(original_gray)
    dft_shift = np.fft.fftshift(dft)
    
    # Reconstruct (Inverse FFT)
    f_ishift = np.fft.ifftshift(dft_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # 3. Inverse Filtering (Wiener Filter simulation) [cite: 25, 31]
    # Deblurring/Denoising approximation
    wiener_out = wiener(original_gray, (5, 5))

    # Visualization
    plt.figure(figsize=(12, 10))
    
    titles = ['Damaged', 'Inpainted', 'Fourier Reconstruction', 'Wiener Filter']
    images = [damaged, inpainted, img_back, wiener_out]
    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        if len(images[i].shape) == 3:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()