#24343031_DEALEXA FATIKA DZIKR_MINGGU 4

import cv2
import numpy as np
import time

class RealTimeEnhancement:
    
    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.history_buffer = []
        self.buffer_size = 5   # untuk temporal consistency
    
    
    def temporal_smoothing(self, frame):
        """Reduce flicker using temporal smoothing"""
        
        self.history_buffer.append(frame)
        
        if len(self.history_buffer) > self.buffer_size:
            self.history_buffer.pop(0)
        
        avg_frame = np.mean(self.history_buffer, axis=0).astype(np.uint8)
        return avg_frame
    
    
    def gamma_correction(self, frame, gamma=0.8):
        """Fast gamma correction"""
        
        invGamma = 1.0 / gamma
        
        table = np.array([
            ((i / 255.0) ** invGamma) * 255
            for i in range(256)
        ]).astype("uint8")
        
        return cv2.LUT(frame, table)
    
    
    def clahe_enhancement(self, frame):
        """Fast CLAHE enhancement"""
        
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l2 = clahe.apply(l)
        
        merged = cv2.merge((l2,a,b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    
    def histogram_equalization(self, frame):
        """Histogram equalization for color image"""
        
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        y_eq = cv2.equalizeHist(y)
        
        merged = cv2.merge((y_eq,cr,cb))
        result = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
        
        return result
    
    
    def adaptive_enhancement(self, frame):
        """Lightweight adaptive enhancement"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        
        if mean_intensity < 100:
            frame = self.gamma_correction(frame, gamma=0.7)
        elif mean_intensity > 180:
            frame = self.gamma_correction(frame, gamma=1.3)
        
        frame = self.clahe_enhancement(frame)
        
        return frame
    
    
    def enhance_frame(self, frame, enhancement_type='adaptive'):
        
        
        if enhancement_type == 'clahe':
            enhanced = self.clahe_enhancement(frame)
        
        elif enhancement_type == 'gamma':
            enhanced = self.gamma_correction(frame)
        
        elif enhancement_type == 'hist_eq':
            enhanced = self.histogram_equalization(frame)
        
        else:
            enhanced = self.adaptive_enhancement(frame)
        
        
        # Temporal consistency
        enhanced = self.temporal_smoothing(enhanced)
        
        return enhanced


# =========================================
# REAL-TIME VIDEO TEST
# =========================================

def run_realtime_enhancement():
    
    cap = cv2.VideoCapture(0)   # webcam
    
    enhancer = RealTimeEnhancement(target_fps=30)
    
    prev_time = 0
    
    while True:
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # resize agar lebih ringan
        frame = cv2.resize(frame,(640,480))
        
        enhanced_frame = enhancer.enhance_frame(frame,'adaptive')
        
        # FPS calculation
        current_time = time.time()
        fps = 1/(current_time-prev_time) if prev_time!=0 else 0
        prev_time = current_time
        
        cv2.putText(enhanced_frame,f"FPS: {fps:.2f}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)
        
        cv2.imshow("Original Video",frame)
        cv2.imshow("Enhanced Video",enhanced_frame)
        
        key = cv2.waitKey(1)
        
        if key == 27:  # ESC untuk keluar
            break
    
    cap.release()
    cv2.destroyAllWindows()


# Run program
run_realtime_enhancement()