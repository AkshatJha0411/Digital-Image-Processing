import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_grid(images, titles):
    plt.figure(figsize=(18, 6))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()

img_path = 'input.jpg'
original = cv2.imread(img_path, 0)

if original is None:
    print("Error: Image not found.")
else:
    # 1. Negative Transformation [cite: 19]
    # Inverts pixel intensities.
    negative = 255 - original

    # 2. Log Transformation [cite: 20]
    # Enhances darker regions: s = c * log(1 + r)
    c = 255 / np.log(1 + np.max(original))
    log_trans = c * (np.log(original + 1))
    log_trans = np.array(log_trans, dtype=np.uint8)

    # 3. Gamma Correction [cite: 20]
    # s = c * r^gamma
    gamma = 2.0 # Gamma > 1 darkens the image
    gamma_trans = np.array(255 * (original / 255) ** gamma, dtype='uint8')

    # 4. Histogram Equalization [cite: 22]
    # Redistributes intensities for better contrast.
    hist_eq = cv2.equalizeHist(original)

    show_grid(
        [original, negative, log_trans, gamma_trans, hist_eq],
        ['Original', 'Negative', 'Log Trans', f'Gamma ({gamma})', 'Hist Equalization']
    )