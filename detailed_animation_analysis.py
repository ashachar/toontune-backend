import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple

def analyze_text_transitions(frame_dir: str):
    """Analyze text animations by comparing consecutive frames"""
    frame_dir = Path(frame_dir)
    
    # Animation segments to analyze (time ranges in seconds)
    segments = [
        (0, 5, "Opening text fade-in"),
        (8, 12, "Multi-line text slide"),
        (18, 22, "Two-line text appearance"),
        (28, 32, "Bottom text with top text"),
        (38, 42, "Text question"),
        (48, 52, "Two-tier text"),
        (58, 62, "Contact info reveal"),
    ]
    
    animations = []
    
    for start_time, end_time, description in segments:
        print(f"\n{'='*60}")
        print(f"Analyzing: {description} ({start_time}s - {end_time}s)")
        print(f"{'='*60}")
        
        # Extract frames for this segment at higher FPS
        segment_frames_path = f"outputs/real_estate_frames/segment_{start_time:03d}_%03d.jpg"
        cmd = f"ffmpeg -i uploads/assets/videos/real_estate.mov -ss {start_time} -t {end_time-start_time} -vf 'fps=15' {segment_frames_path} -y"
        
        import subprocess
        subprocess.run(cmd, shell=True, capture_output=True)
        
        # Analyze frames in this segment
        segment_frames = sorted(Path("outputs/real_estate_frames").glob(f"segment_{start_time:03d}_*.jpg"))
        
        if not segment_frames:
            print(f"  No frames extracted for segment")
            continue
            
        print(f"  Analyzing {len(segment_frames)} frames...")
        
        # Analyze first, middle and last frames
        key_frame_indices = [0, len(segment_frames)//4, len(segment_frames)//2, 3*len(segment_frames)//4, -1]
        
        text_regions = []
        prev_gray = None
        
        for idx in key_frame_indices:
            if idx >= len(segment_frames) or idx < -len(segment_frames):
                continue
                
            frame = cv2.imread(str(segment_frames[idx]))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect text regions using thresholding
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter for text-like contours
            frame_text_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 50000:  # Text size range
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.5 < aspect_ratio < 20:  # Text aspect ratio
                        frame_text_regions.append({
                            'x': int(x), 
                            'y': int(y), 
                            'width': int(w), 
                            'height': int(h),
                            'area': int(area),
                            'frame_idx': int(idx if idx >= 0 else len(segment_frames) + idx)
                        })
            
            text_regions.append(frame_text_regions)
            
            # Detect motion/changes
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                motion_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
                motion_area = np.sum(motion_mask > 0)
                motion_percentage = (motion_area / (gray.shape[0] * gray.shape[1])) * 100
                print(f"    Frame {idx}: Motion {motion_percentage:.1f}%, Text regions: {len(frame_text_regions)}")
            
            prev_gray = gray
        
        # Determine animation type based on analysis
        animation_type = detect_animation_type(text_regions, segment_frames, start_time)
        
        animations.append({
            'start_time': start_time,
            'end_time': end_time,
            'description': description,
            'animation_type': animation_type,
            'frame_count': len(segment_frames)
        })
        
        print(f"\n  Detected animation type: {animation_type}")
    
    return animations

def detect_animation_type(text_regions: List, frames: List, start_time: int) -> str:
    """Detect the type of text animation based on region changes"""
    
    if not text_regions or not text_regions[0]:
        return "no_text_detected"
    
    # Check if text appears gradually (fade in)
    if len(text_regions[0]) == 0 and len(text_regions[-1]) > 0:
        # Check if it's sliding in from a direction
        if text_regions[-1][0]['y'] < frames[0].shape[0] // 3:
            return "fade_in_top"
        elif text_regions[-1][0]['y'] > 2 * frames[0].shape[0] // 3:
            return "fade_in_bottom"
        else:
            return "fade_in_center"
    
    # Check for multi-line sequential appearance
    if len(text_regions) > 2:
        region_counts = [len(r) for r in text_regions]
        if region_counts[-1] > region_counts[0]:
            return "sequential_line_reveal"
    
    # Check for position changes (sliding)
    if len(text_regions[0]) > 0 and len(text_regions[-1]) > 0:
        y_change = abs(text_regions[-1][0]['y'] - text_regions[0][0]['y'])
        if y_change > 20:
            return "vertical_slide"
    
    return "static_text"

def create_animation_summary(animations: List[Dict]) -> str:
    """Create a summary of detected animations"""
    
    # Group animations by type
    animation_groups = {}
    for anim in animations:
        anim_type = anim['animation_type']
        if anim_type not in animation_groups:
            animation_groups[anim_type] = []
        animation_groups[anim_type].append(anim)
    
    summary = "# Text Animation Analysis Summary\n\n"
    summary += "## Animation Types Detected:\n\n"
    
    for anim_type, instances in animation_groups.items():
        summary += f"### {anim_type.replace('_', ' ').title()}\n"
        summary += f"- **Count**: {len(instances)} instances\n"
        summary += f"- **Occurrences**:\n"
        for inst in instances:
            summary += f"  - {inst['description']} ({inst['start_time']}s - {inst['end_time']}s)\n"
        summary += "\n"
    
    return summary

if __name__ == "__main__":
    print("Starting detailed text animation analysis...")
    print("=" * 60)
    
    animations = analyze_text_transitions("outputs/real_estate_frames")
    
    print("\n" + "=" * 60)
    print("ANIMATION SUMMARY")
    print("=" * 60)
    
    summary = create_animation_summary(animations)
    print(summary)
    
    # Save summary to file
    with open("outputs/animation_analysis_summary.md", "w") as f:
        f.write(summary)
    
    print(f"\nSummary saved to: outputs/animation_analysis_summary.md")
    
    # Save detailed JSON
    with open("outputs/animation_analysis_detailed.json", "w") as f:
        json.dump(animations, f, indent=2)
    
    print(f"Detailed analysis saved to: outputs/animation_analysis_detailed.json")