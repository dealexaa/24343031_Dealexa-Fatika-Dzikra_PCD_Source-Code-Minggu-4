#24343031_DEALEXA FATIKA DZIKR_MINGGU 4

import cv2
import numpy as np
from scipy import stats

def calculate_metrics(original, enhanced):
    """Calculate image enhancement metrics"""
    
    metrics = {}
    
    # Mean dan standard deviation
    metrics['original_mean'] = np.mean(original)
    metrics['enhanced_mean'] = np.mean(enhanced)
    
    metrics['original_std'] = np.std(original)
    metrics['enhanced_std'] = np.std(enhanced)
    
    # Contrast Improvement Index
    if metrics['original_std'] != 0:
        metrics['CII'] = metrics['enhanced_std'] / metrics['original_std']
    else:
        metrics['CII'] = 0
    
    # Entropy
    hist_orig,_ = np.histogram(original.flatten(),256,[0,256])
    hist_enh,_ = np.histogram(enhanced.flatten(),256,[0,256])
    
    metrics['original_entropy'] = stats.entropy(hist_orig+1e-10)
    metrics['enhanced_entropy'] = stats.entropy(hist_enh+1e-10)
    
    metrics['entropy_gain'] = metrics['enhanced_entropy'] - metrics['original_entropy']
    
    return metrics


def medical_image_enhancement(medical_image, modality='X-ray'):
    

    image = medical_image.copy()
    
    # Step 1: Noise Reduction
    if modality == 'Ultrasound':
        denoised = cv2.medianBlur(image,5)  # speckle noise
    else:
        denoised = cv2.GaussianBlur(image,(3,3),0)
    
    # Step 2: Modality specific enhancement
    if modality == 'X-ray':
        
        # Gamma correction (improve bone contrast)
        gamma = 0.7
        norm = image / 255.0
        gamma_corrected = np.power(norm,gamma)
        gamma_corrected = np.uint8(gamma_corrected*255)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
        enhanced = clahe.apply(gamma_corrected)
    
    elif modality == 'MRI':
        
        # Contrast stretching
        min_val = np.min(denoised)
        max_val = np.max(denoised)
        stretched = ((denoised-min_val)/(max_val-min_val)*255).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        enhanced = clahe.apply(stretched)
    
    elif modality == 'CT':
        
        # Windowing simulation
        window_center = np.mean(denoised)
        window_width = np.std(denoised)*4
        
        low = window_center-window_width/2
        high = window_center+window_width/2
        
        windowed = np.clip(denoised,low,high)
        windowed = ((windowed-low)/(high-low)*255).astype(np.uint8)
        
        enhanced = windowed
    
    elif modality == 'Ultrasound':
        
        clahe = cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
    
    else:
        enhanced = denoised
    
    # Step 3: Calculate metrics
    metrics = calculate_metrics(image, enhanced)
    
    # Step 4: Generate report
    report = {
        "modality": modality,
        "processing_steps":[
            "Noise Reduction",
            "Modality Specific Enhancement",
            "Contrast Improvement"
        ],
        "metrics": metrics
    }
    
    return enhanced, report


# ===============================
# TEST PROGRAM
# ===============================

import matplotlib.pyplot as plt

# contoh gambar (simulasi medical image)
medical_img = np.random.normal(120,30,(256,256))
medical_img = np.clip(medical_img,0,255).astype(np.uint8)

enhanced_img, report = medical_image_enhancement(medical_img,'X-ray')

# tampilkan hasil
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(medical_img,cmap='gray')
plt.title("Original Medical Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(enhanced_img,cmap='gray')
plt.title("Enhanced Image")
plt.axis('off')

plt.show()

# tampilkan report
print("\n=== ENHANCEMENT REPORT ===")
print("Modality :", report["modality"])
print("\nProcessing Steps:")
for step in report["processing_steps"]:
    print("-",step)

print("\nMetrics:")
for k,v in report["metrics"].items():
    print(f"{k} : {v:.4f}")