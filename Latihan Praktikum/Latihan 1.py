#24343031_DEALEXA FATIKA DZIKR_MINGGU 4

import numpy as np
import cv2
import matplotlib.pyplot as plt

def manual_histogram_equalization(image):
    """
    Manual implementation of histogram equalization
    
    Parameters:
    image: Input grayscale image (0-255)
    
    Returns:
    equalized_image: Hasil citra setelah equalization
    transform: Fungsi transformasi (lookup table)
    """

    # 1. Hitung histogram
    histogram = np.zeros(256)
    for pixel in image.flatten():
        histogram[pixel] += 1

    # 2. Hitung cumulative histogram (CDF)
    cumulative_histogram = np.cumsum(histogram)

    # 3. Hitung transformation function
    # Normalisasi CDF ke rentang 0-255
    cdf_min = cumulative_histogram[np.nonzero(cumulative_histogram)].min()
    total_pixels = image.size

    transform = ((cumulative_histogram - cdf_min) / (total_pixels - cdf_min)) * 255
    transform = np.round(transform).astype(np.uint8)

    # 4. Apply transformation
    equalized_image = transform[image]

    # 5. Return equalized image dan transformation function
    return equalized_image, transform


# ================================
# TEST PROGRAM
# ================================

# Load gambar grayscale
image = cv2.imread("Latihan Praktikum/image.png", cv2.IMREAD_GRAYSCALE)

# Jalankan histogram equalization manual
equalized_img, transform_func = manual_histogram_equalization(image)

# Visualisasi
plt.figure(figsize=(12,6))

plt.subplot(2,3,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,3,2)
plt.hist(image.ravel(),256,[0,256])
plt.title("Original Histogram")

plt.subplot(2,3,4)
plt.imshow(equalized_img, cmap='gray')
plt.title("Equalized Image (Manual)")
plt.axis("off")

plt.subplot(2,3,5)
plt.hist(equalized_img.ravel(),256,[0,256])
plt.title("Equalized Histogram")

plt.subplot(2,3,3)
plt.plot(transform_func)
plt.title("Transformation Function")
plt.xlabel("Input Intensity")
plt.ylabel("Output Intensity")

plt.tight_layout()
plt.show()