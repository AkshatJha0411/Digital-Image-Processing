import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_comparison(img_list, title_list):
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(img_list, title_list)):
        plt.subplot(1, len(img_list), i+1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(title)
        plt.axis('off')
    plt.show()

# Load image in grayscale
img_path = 'input.jpg'  # Replace with your image name
original = cv2.imread(img_path, 0)

if original is None:
    print("Error: Image not found.")
else:
    # 1. Sampling: Reduce resolution by skipping pixels (slicing)
    # [cite: 4] Sampling determines spatial resolution.
    k = 4  # Sampling factor
    sampled = original[::k, ::k]

    # 2. Quantization: Reduce intensity levels
    # [cite: 6] Quantization maps continuous intensity to finite levels.
    # Reducing 8-bit (256 levels) to 3-bit (8 levels)
    levels = 8
    div = 256 / levels
    quantized = np.floor(original / div) * div
    quantized = quantized.astype(np.uint8)

    print(f"Sampling factor: {k}, Quantization levels: {levels}")
    show_comparison([original, sampled, quantized], 
                    ['Original', f'Sampled (1/{k})', f'Quantized ({levels} Levels)'])