#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to visualize face detection in all frames.
Shows detected face regions with bounding boxes.
"""

import cv2
import numpy as np
import subprocess
import os

class FaceDetector:
    """Detects faces in frames using OpenCV's Haar Cascade"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_faces(self, frame: np.ndarray):
        """Detect faces in a frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Same parameters as in per_phrase_sandwich.py
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(50, 50)
        )
        
        face_regions = []
        for (x, y, w, h) in faces:
            # Add 20% margin (same as in original)
            margin = int(w * 0.2)
            x_expanded = max(0, x - margin)
            y_expanded = max(0, y - margin)
            w_expanded = w + (margin * 2)
            h_expanded = h + (margin * 2)
            
            face_regions.append({
                'original': (x, y, w, h),
                'expanded': (x_expanded, y_expanded, w_expanded, h_expanded)
            })
        
        return face_regions


def create_face_detection_debug_video(input_video: str, output_path: str):
    """Create a debug video showing face detection in all frames"""
    
    # Initialize face detector
    face_detector = FaceDetector()
    
    # Open video
    cap = cv2.VideoCapture(input_video)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing {total_frames} frames for face detection visualization...")
    
    face_count_stats = []
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        face_regions = face_detector.detect_faces(frame)
        
        # Draw face regions
        debug_frame = frame.copy()
        
        for face in face_regions:
            # Draw original detection in green
            x, y, w, h = face['original']
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(debug_frame, "Original", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw expanded region in red (this is what's used for overlap)
            x_exp, y_exp, w_exp, h_exp = face['expanded']
            cv2.rectangle(debug_frame, (x_exp, y_exp), (x_exp + w_exp, y_exp + h_exp), 
                         (0, 0, 255), 2)
            cv2.putText(debug_frame, "Expanded (+20%)", (x_exp, y_exp - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Add frame info
        info_text = f"Frame {frame_idx}/{total_frames} | Faces: {len(face_regions)}"
        cv2.putText(debug_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(debug_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        # Add legend
        cv2.putText(debug_frame, "Green = Original Detection", (10, height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(debug_frame, "Red = Expanded Region (used for overlap)", (10, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        out.write(debug_frame)
        
        face_count_stats.append(len(face_regions))
        
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
            if len(face_regions) > 0:
                print(f"  Detected {len(face_regions)} face(s)")
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nFace detection statistics:")
    print(f"  Total frames: {len(face_count_stats)}")
    print(f"  Frames with faces: {sum(1 for c in face_count_stats if c > 0)}")
    print(f"  Frames without faces: {sum(1 for c in face_count_stats if c == 0)}")
    print(f"  Average faces per frame: {sum(face_count_stats) / len(face_count_stats):.2f}")
    
    print(f"\nDebug video saved: {output_path}")
    
    # Convert to H.264
    output_h264 = output_path.replace('.mp4', '_h264.mp4')
    cmd = [
        "ffmpeg", "-y",
        "-i", output_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_h264
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"H.264 version: {output_h264}")
    
    # Remove temp file
    os.remove(output_path)
    return output_h264


def main():
    input_video = "ai_math1_6sec.mp4"
    output_video = "face_detection_debug.mp4"
    
    final_video = create_face_detection_debug_video(input_video, output_video)
    
    print(f"\n✅ Face detection debug video created: {final_video}")
    print("\nVisualization shows:")
    print("  • Green boxes: Original face detection")
    print("  • Red boxes: Expanded regions (+20% margin)")
    print("  • Frame counter and face count")
    print("  • The red boxes are what's used for text overlap checking")


if __name__ == "__main__":
    main()