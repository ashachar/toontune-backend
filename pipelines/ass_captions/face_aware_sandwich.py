#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face-aware sandwich compositing for ASS captions.
Only applies text-behind effect when text overlaps with detected faces.
"""

import cv2
import numpy as np
import subprocess
import os
import json
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class FaceRegion:
    """Represents a detected face region"""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def left(self) -> int:
        return self.x
    
    @property
    def right(self) -> int:
        return self.x + self.width
    
    @property
    def top(self) -> int:
        return self.y
    
    @property
    def bottom(self) -> int:
        return self.y + self.height


class FaceDetector:
    """Detects faces in frames using OpenCV's Haar Cascade"""
    
    def __init__(self):
        # Initialize face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self._face_cache = {}
    
    def detect_faces(self, frame: np.ndarray, frame_idx: int) -> List[FaceRegion]:
        """Detect faces in a frame"""
        # Check cache
        if frame_idx in self._face_cache:
            return self._face_cache[frame_idx]
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(50, 50)
        )
        
        # Convert to FaceRegion objects with margin
        face_regions = []
        for (x, y, w, h) in faces:
            # Add 20% margin around face
            margin = int(w * 0.2)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = w + (margin * 2)
            h = h + (margin * 2)
            
            face_regions.append(FaceRegion(x=x, y=y, width=w, height=h))
        
        # Cache results
        self._face_cache[frame_idx] = face_regions
        return face_regions


def get_text_regions_at_time(transcript_data: Dict, current_time: float) -> List[Dict]:
    """
    Get text regions that should be visible at current time.
    Returns list of dicts with text, position, and importance info.
    """
    visible_texts = []
    
    for phrase in transcript_data.get("phrases", []):
        if phrase["start_time"] <= current_time <= phrase["end_time"]:
            # Estimate text position based on phrase properties
            # This is approximate - actual ASS rendering may differ slightly
            font_size = int(48 * phrase["visual_style"]["font_size_multiplier"])
            
            # Estimate position
            if phrase["position"] == "top":
                y = 180  # 25% from top for 720p
            else:
                y = 540  # 25% from bottom for 720p
            
            # Estimate width (rough approximation)
            text = phrase["text"]
            estimated_width = len(text) * int(font_size * 0.6)
            x = (1280 - estimated_width) // 2  # Centered
            
            visible_texts.append({
                "text": text,
                "x": x,
                "y": y - font_size,  # Top edge of text
                "width": estimated_width,
                "height": int(font_size * 1.5),
                "importance": phrase["importance"],
                "emphasis_type": phrase["emphasis_type"]
            })
    
    return visible_texts


def check_text_face_overlap(text_region: Dict, face_regions: List[FaceRegion]) -> bool:
    """
    Check if text region overlaps with any face.
    """
    text_left = text_region["x"]
    text_right = text_region["x"] + text_region["width"]
    text_top = text_region["y"]
    text_bottom = text_region["y"] + text_region["height"]
    
    for face in face_regions:
        # Check for overlap
        if (text_left < face.right and text_right > face.left and
            text_top < face.bottom and text_bottom > face.top):
            return True
    
    return False


def burn_ass_to_video(input_video: str, ass_file: str, temp_output: str):
    """Burn ASS subtitles into video using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", f"subtitles={ass_file}",
        "-c:v", "libx264",
        "-preset", "fast", 
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        temp_output
    ]
    
    print(f"Burning ASS subtitles to video...")
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Created video with burned subtitles: {temp_output}")


def apply_face_aware_sandwich(
    video_with_text: str,
    original_video: str, 
    mask_video: str,
    transcript_path: str,
    output_path: str
):
    """
    Apply foreground masking only when text overlaps with faces.
    Pre-analyzes entire text duration to ensure consistent behavior.
    """
    # Load transcript for text position information
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    # Initialize face detector
    face_detector = FaceDetector()
    
    # Open videos for pre-analysis
    cap_orig = cv2.VideoCapture(original_video)
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("Pre-analyzing text-face overlaps for consistent masking decisions...")
    
    # Pre-analyze which phrases should be behind
    phrases_should_be_behind = set()
    
    for phrase in transcript_data.get("phrases", []):
        start_frame = int(phrase["start_time"] * fps)
        end_frame = int(phrase["end_time"] * fps)
        
        # Sample frames during this phrase's display period
        # Check every 5 frames for efficiency
        face_overlap_detected = False
        
        for check_frame in range(start_frame, min(end_frame + 1, total_frames), 5):
            cap_orig.set(cv2.CAP_PROP_POS_FRAMES, check_frame)
            ret, frame = cap_orig.read()
            if not ret:
                continue
                
            # Detect faces
            face_regions = face_detector.detect_faces(frame, check_frame)
            
            if len(face_regions) > 0:
                # Get text region for this phrase
                font_size = int(48 * phrase["visual_style"]["font_size_multiplier"])
                
                if phrase["position"] == "top":
                    y = 180
                else:
                    y = 540
                
                text = phrase["text"]
                estimated_width = len(text) * int(font_size * 0.6)
                x = (1280 - estimated_width) // 2
                
                text_region = {
                    "x": x,
                    "y": y - font_size,
                    "width": estimated_width,
                    "height": int(font_size * 1.5)
                }
                
                # Check overlap
                if check_text_face_overlap(text_region, face_regions):
                    face_overlap_detected = True
                    break
        
        if face_overlap_detected:
            # Mark this phrase to always be behind
            phrase_key = f"{phrase['start_time']:.2f}_{phrase['text'][:20]}"
            phrases_should_be_behind.add(phrase_key)
            print(f"  Phrase '{phrase['text'][:30]}...' will be behind (face detected)")
    
    cap_orig.release()
    
    print(f"Marked {len(phrases_should_be_behind)} phrases for text-behind effect")
    
    # Now process video with consistent decisions
    cap_text = cv2.VideoCapture(video_with_text)
    cap_orig = cv2.VideoCapture(original_video)
    cap_mask = cv2.VideoCapture(mask_video)
    
    # Get video properties
    fps = cap_text.get(cv2.CAP_PROP_FPS)
    width = int(cap_text.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_text.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing {total_frames} frames with consistent masking...")
    
    for frame_idx in range(total_frames):
        ret_text, frame_with_text = cap_text.read()
        ret_orig, frame_original = cap_orig.read()
        ret_mask, mask_frame = cap_mask.read()
        
        if not ret_text or not ret_orig or not ret_mask:
            break
        
        current_time = frame_idx / fps
        
        # Check if current text should be behind
        apply_masking = False
        for phrase in transcript_data.get("phrases", []):
            if phrase["start_time"] <= current_time <= phrase["end_time"]:
                phrase_key = f"{phrase['start_time']:.2f}_{phrase['text'][:20]}"
                if phrase_key in phrases_should_be_behind:
                    apply_masking = True
                    break
        
        if apply_masking:
            # Apply text-behind effect for this frame
            # Green screen detection with tolerance
            green_screen_color = np.array([154, 254, 119], dtype=np.uint8)
            tolerance = 25
            
            diff = np.abs(mask_frame.astype(np.int16) - green_screen_color.astype(np.int16))
            is_green_screen = np.all(diff <= tolerance, axis=2)
            
            # Person mask: NOT green screen
            person_mask = (~is_green_screen).astype(np.uint8)
            
            # Light erosion to allow text closer to edges
            kernel = np.ones((2,2), np.uint8)
            person_mask = cv2.erode(person_mask, kernel, iterations=1)
            
            # Convert to 3-channel
            person_mask_3ch = np.stack([person_mask, person_mask, person_mask], axis=2)
            
            # Composite: original where person is, text where background
            final_frame = np.where(person_mask_3ch == 1, frame_original, frame_with_text)
        else:
            # No face overlap - show text normally (on top)
            final_frame = frame_with_text
        
        final_frame = final_frame.astype(np.uint8)
        out.write(final_frame)
        
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")
            if apply_masking:
                print(f"  ✓ Text-behind effect applied (consistent for phrase)")
    
    # Clean up
    cap_text.release()
    cap_orig.release()
    cap_mask.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video saved: {output_path}")
    
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
    # Step 1: Burn ASS subtitles to video
    input_video = "ai_math1_6sec.mp4"
    ass_file = "ai_math1_wordbyword_captions.ass"
    temp_video_with_text = "temp_with_text.mp4"
    
    burn_ass_to_video(input_video, ass_file, temp_video_with_text)
    
    # Step 2: Apply face-aware sandwich compositing
    mask_video = "../../uploads/assets/videos/ai_math1/ai_math1_rvm_mask.mp4"
    transcript_path = "../../uploads/assets/videos/ai_math1/transcript_enriched_partial.json"
    output_video = "ai_math1_face_aware_sandwich.mp4"
    
    final_video = apply_face_aware_sandwich(
        temp_video_with_text,
        input_video,
        mask_video,
        transcript_path,
        output_video
    )
    
    # Clean up temp file
    os.remove(temp_video_with_text)
    
    print(f"\n✅ Face-aware video created: {final_video}")
    print("\nFeatures:")
    print("  • Face detection using OpenCV Haar Cascade")
    print("  • Pre-analyzes entire video for consistent decisions")
    print("  • Text-behind effect for ENTIRE phrase duration if ANY overlap detected")
    print("  • No flickering - each phrase is consistently front or behind")
    print("  • Automatic face region expansion for safety margin")


if __name__ == "__main__":
    main()