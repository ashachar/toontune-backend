import cv2
import numpy as np
from PIL import Image
import sys

def sketch_to_thin_lines(input_path, output_path):
    """Convert Anime2Sketch output to thin binary lines"""
    
    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    # 1. Threshold to get binary image (remove grays)
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    
    # 2. Invert (make lines black, background white)
    binary = 255 - binary
    
    # 3. Apply morphological thinning (skeletonization)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    thin = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 4. Use Zhang-Suen thinning algorithm for 1-pixel lines
    thin = cv2.ximgproc.thinning(thin, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    
    # 5. Invert back (black lines on white background)
    result = 255 - thin
    
    cv2.imwrite(output_path, result)
    return result

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_to_thin_lines.py input.jpg output.png")
        sys.exit(1)
    
    sketch_to_thin_lines(sys.argv[1], sys.argv[2])
    print(f"Thin lines saved to {sys.argv[2]}")