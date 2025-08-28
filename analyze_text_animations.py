import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple

def analyze_frame_sequence(frame_dir: str) -> Dict:
    """Analyze sequence of frames to detect text animations"""
    frame_dir = Path(frame_dir)
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    
    if not frames:
        print(f"No frames found in {frame_dir}")
        return {}
    
    animations_detected = []
    prev_frame = None
    
    # Key frames to analyze in detail
    key_frames = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120, 130, 140, 145]
    
    print(f"Analyzing {len(frames)} frames...")
    
    for idx in key_frames:
        if idx >= len(frames):
            continue
            
        frame_path = frames[idx]
        frame = cv2.imread(str(frame_path))
        frame_time = idx  # seconds
        
        print(f"\n=== Frame {idx:03d} (t={frame_time}s) ===")
        print(f"  Path: {frame_path.name}")
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect edges (potential text regions)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours (potential text boxes)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for text-like regions
        text_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100 and area < 10000:  # Filter by size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.5 < aspect_ratio < 10:  # Text typically has certain aspect ratios
                    text_regions.append((x, y, w, h))
        
        # Analyze frame characteristics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Detect motion by comparing with previous frame
        motion_detected = False
        if prev_frame is not None and idx > 0:
            diff = cv2.absdiff(prev_frame, gray)
            motion_score = np.mean(diff)
            motion_detected = motion_score > 10
            print(f"  Motion Score: {motion_score:.2f}")
        
        print(f"  Brightness: mean={mean_brightness:.1f}, std={std_brightness:.1f}")
        print(f"  Potential text regions: {len(text_regions)}")
        print(f"  Motion detected: {motion_detected}")
        
        # Detect dominant colors (for text color analysis)
        # Reshape frame to list of pixels
        pixels = frame.reshape(-1, 3)
        # Use k-means to find dominant colors
        from sklearn.cluster import KMeans
        n_colors = 5
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        
        print(f"  Dominant colors (BGR): {colors.tolist()}")
        
        # Check for specific regions of the frame
        height, width = frame.shape[:2]
        
        # Top region (often titles)
        top_region = frame[0:height//4, :]
        top_brightness = np.mean(cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY))
        
        # Bottom region (often subtitles/captions)
        bottom_region = frame[3*height//4:, :]
        bottom_brightness = np.mean(cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY))
        
        # Center region
        center_region = frame[height//4:3*height//4, width//4:3*width//4]
        center_brightness = np.mean(cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY))
        
        print(f"  Region brightness - Top: {top_brightness:.1f}, Center: {center_brightness:.1f}, Bottom: {bottom_brightness:.1f}")
        
        # Save analysis image for key frames
        if idx in [0, 20, 40, 60, 80, 100, 120, 145]:
            analysis_frame = frame.copy()
            
            # Draw detected text regions
            for (x, y, w, h) in text_regions[:5]:  # Show top 5 regions
                cv2.rectangle(analysis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Save annotated frame
            output_path = f"outputs/real_estate_frames/analysis_frame_{idx:04d}.jpg"
            cv2.imwrite(output_path, analysis_frame)
            print(f"  Saved analysis frame to: {output_path}")
        
        prev_frame = gray
    
    return {
        "total_frames": len(frames),
        "analyzed_frames": len(key_frames),
        "video_duration_seconds": len(frames),
    }

if __name__ == "__main__":
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("Installing scikit-learn...")
        import subprocess
        subprocess.check_call(["pip", "install", "scikit-learn"])
        from sklearn.cluster import KMeans
    
    print("Starting real_estate.mov text animation analysis...")
    print("=" * 60)
    
    results = analyze_frame_sequence("outputs/real_estate_frames")
    
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2))