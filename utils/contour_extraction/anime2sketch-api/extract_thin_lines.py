import cv2
import numpy as np
from skimage.morphology import skeletonize
from PIL import Image
import sys

def extract_thin_contours(input_path, output_path, threshold=180):
    """Extract thin contour lines from sketch image"""
    
    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    # 1. Strong threshold to get only dark lines
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    # 2. Find edges using Canny
    edges = cv2.Canny(binary, 50, 150)
    
    # 3. Optional: Dilate then erode to connect broken lines
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 4. Thin the edges to 1-pixel width using skeletonization
    # Convert to binary format for skimage (0 and 1)
    binary_for_skel = edges > 0
    
    # Apply skeletonization
    skeleton = skeletonize(binary_for_skel)
    
    # Convert back to 0-255 range
    result = (skeleton * 255).astype(np.uint8)
    
    # Invert if needed (black lines on white background)
    result = 255 - result
    
    cv2.imwrite(output_path, result)
    print(f"Saved thin lines to {output_path}")
    
    # Also save intermediate steps for debugging
    cv2.imwrite(output_path.replace('.png', '_edges.png'), edges)
    cv2.imwrite(output_path.replace('.png', '_binary.png'), binary)
    
    return result

def simple_thin_lines(input_path, output_path):
    """Simpler approach - just threshold and edge detection"""
    
    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Use adaptive threshold for better line extraction
    binary = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(255-binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create output image
    result = np.ones_like(img) * 255
    
    # Draw contours with thin lines
    cv2.drawContours(result, contours, -1, (0, 0, 0), 1)
    
    cv2.imwrite(output_path, result)
    print(f"Saved simple thin lines to {output_path}")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_thin_lines.py input.jpg output.png [method]")
        print("Methods: skeleton (default), simple")
        sys.exit(1)
    
    method = sys.argv[3] if len(sys.argv) > 3 else "skeleton"
    
    if method == "simple":
        simple_thin_lines(sys.argv[1], sys.argv[2])
    else:
        try:
            extract_thin_contours(sys.argv[1], sys.argv[2])
        except ImportError:
            print("Skeletonize not available, using simple method")
            simple_thin_lines(sys.argv[1], sys.argv[2])