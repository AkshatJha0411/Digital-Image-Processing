import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = 'input.jpg'
gray_img = cv2.imread(img_path, 0)
color_img = cv2.imread(img_path)

if gray_img is None:
    print("Error: Image not found.")
else:
    # 1. Spatial Resolution Analysis
    # [cite: 14] Generates 2x and 4x lower resolutions to visualize detail loss.
    h, w = gray_img.shape
    res_2x = cv2.resize(gray_img, (w//2, h//2))
    res_4x = cv2.resize(gray_img, (w//4, h//4))

    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(gray_img, cmap='gray'), plt.title('Original'), plt.axis('off')
    plt.subplot(132), plt.imshow(res_2x, cmap='gray'), plt.title('Resolution / 2'), plt.axis('off')
    plt.subplot(133), plt.imshow(res_4x, cmap='gray'), plt.title('Resolution / 4'), plt.axis('off')
    plt.show()

    # 2. Histogram of Color Channels
    # [cite: 16] Shows intensity distribution for R, G, B channels.
    colors = ('b', 'g', 'r')
    plt.figure(figsize=(10, 5))
    plt.title('Histogram of Color Channels')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([color_img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    
    plt.xlim([0, 256])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.show()