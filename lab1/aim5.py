import cv2
import matplotlib.pyplot as plt

img_path = 'input.jpg'
original = cv2.imread(img_path)

if original is None:
    print("Error: Image not found.")
else:
    # Decimation (Downsampling) [cite: 37]
    # Reduce size to 25%
    small = cv2.resize(original, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    # Interpolation (Upscaling) [cite: 34]
    # 1. Nearest Neighbor
    nearest = cv2.resize(small, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    
    # 2. Bilinear Interpolation
    bilinear = cv2.resize(small, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    
    # 3. Bicubic Interpolation (Smoothest)
    bicubic = cv2.resize(small, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Display results
    images = [small, nearest, bilinear, bicubic]
    titles = ['Decimated (Small)', 'Nearest Neighbor', 'Bilinear', 'Bicubic']

    plt.figure(figsize=(15, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        # Convert BGR to RGB for matplotlib
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.show()