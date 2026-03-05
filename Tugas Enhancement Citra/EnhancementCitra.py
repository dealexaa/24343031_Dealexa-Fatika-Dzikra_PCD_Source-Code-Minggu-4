#24343031_DEALEXA FATIKA DZIKR_MINGGU 4

# ==========================================
# IMAGE ENHANCEMENT ANALYSIS PROGRAM
# ==========================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# ==========================================
# LOAD IMAGE
# ==========================================

underexposed = cv2.imread("Tugas Enhancement Citra/dark.jpeg", cv2.IMREAD_GRAYSCALE)
overexposed = cv2.imread("Tugas Enhancement Citra/bright.jpeg", cv2.IMREAD_GRAYSCALE)
uneven = cv2.imread("Tugas Enhancement Citra/uneven.jpeg", cv2.IMREAD_GRAYSCALE)

images = {
    "Underexposed": underexposed,
    "Overexposed": overexposed,
    "Uneven Illumination": uneven
}


# ==========================================
# POINT PROCESSING
# ==========================================

def negative_transform(img):
    return 255 - img


def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    log_img = c * np.log(1 + img.astype(np.float32))
    return np.array(log_img, dtype=np.uint8)


def gamma_transform(img, gamma):
    norm = img / 255.0
    gamma_img = np.power(norm, gamma)
    return np.uint8(gamma_img * 255)


# ==========================================
# HISTOGRAM BASED ENHANCEMENT
# ==========================================

def contrast_stretch_manual(img, rmin, rmax):

    stretched = (img - rmin) / (rmax - rmin) * 255
    stretched = np.clip(stretched, 0, 255)

    return stretched.astype(np.uint8)


def contrast_stretch_auto(img):

    rmin = np.min(img)
    rmax = np.max(img)

    return contrast_stretch_manual(img, rmin, rmax)


def histogram_equalization(img):

    return cv2.equalizeHist(img)


def clahe_enhancement(img):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)


# ==========================================
# METRICS
# ==========================================

def contrast_ratio(img):
    return np.std(img)


def entropy_metric(img):

    hist,_ = np.histogram(img.flatten(),256,[0,256])
    return entropy(hist + 1e-10)


def snr(img):

    signal = np.mean(img)
    noise = np.std(img)

    if noise == 0:
        return 0

    return signal / noise


def evaluate_metrics(name, original, enhanced):

    print("\n----", name, "----")

    print("Contrast:", contrast_ratio(enhanced))
    print("Entropy :", entropy_metric(enhanced))
    print("SNR     :", snr(enhanced))


# ==========================================
# VISUALIZATION
# ==========================================

def show_results(title, original, enhanced):

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(enhanced, cmap='gray')
    plt.title(title)
    plt.axis("off")

    plt.show()


def show_histograms(original, enhanced):

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.hist(original.ravel(),256,[0,256])
    plt.title("Original Histogram")

    plt.subplot(1,2,2)
    plt.hist(enhanced.ravel(),256,[0,256])
    plt.title("Enhanced Histogram")

    plt.show()


# ==========================================
# PROCESS ALL IMAGES
# ==========================================

for name, img in images.items():

    print("\n=================================")
    print("Processing:", name)
    print("=================================")

    # POINT PROCESSING
    neg = negative_transform(img)
    log = log_transform(img)

    gamma1 = gamma_transform(img, 0.5)
    gamma2 = gamma_transform(img, 1.5)
    gamma3 = gamma_transform(img, 2.5)

    # HISTOGRAM ENHANCEMENT
    stretch_auto = contrast_stretch_auto(img)
    stretch_manual = contrast_stretch_manual(img, 50, 200)

    hist_eq = histogram_equalization(img)
    clahe = clahe_enhancement(img)

    # Visual inspection
    show_results("Negative", img, neg)
    show_results("Log Transform", img, log)

    show_results("Gamma 0.5", img, gamma1)
    show_results("Gamma 1.5", img, gamma2)
    show_results("Gamma 2.5", img, gamma3)

    show_results("Contrast Stretch Auto", img, stretch_auto)
    show_results("Contrast Stretch Manual", img, stretch_manual)

    show_results("Histogram Equalization", img, hist_eq)
    show_results("CLAHE", img, clahe)

    # Histogram comparison
    show_histograms(img, hist_eq)
    show_histograms(img, clahe)

    # Quantitative metrics
    evaluate_metrics("Histogram Equalization", img, hist_eq)
    evaluate_metrics("CLAHE", img, clahe)