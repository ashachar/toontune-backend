#!/usr/bin/env python3
"""
Semantic 3D Word Animation Pipeline
Built on complete_text_animation_pipeline.py with WordRiseSequence3D
Adds semantic understanding and visual hierarchy
"""

import cv2
import numpy as np
import sys
import os
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Add animation modules to path
sys.path.append('utils/animations')
sys.path.append('utils/animations/3d_animations')
sys.path.append('utils/animations/3d_animations/word_3d')

# Import 3D animation components
from base_3d_text_animation import Animation3DConfig
from word_3d import WordRiseSequence3D
from utils.animations.fog_dissolve import CleanFogDissolve
from utils.transcript.enrich_transcript import TranscriptEnricher

@dataclass
class SemanticWord3D:
    """Word with semantic properties for 3D animation"""
    text: str
    start_time: float
    end_time: float
    importance: float
    emphasis_type: str
    appearance_index: int  # Scene number
    layout_priority: int
    # Visual properties from emphasis
    font_size_mult: float
    color: Tuple[int, int, int]
    bold: bool
    glow: bool

class Semantic3DWordPipeline:
    """Semantic word animation using WordRiseSequence3D"""
    
    def __init__(self):
        self.base_font_size = 48
        self.rise_duration = 0.8
        self.dissolve_duration = 1.5
        
        # Visual emphasis settings
        self.emphasis_styles = {
            "mega_title": {
                "size_mult": 2.2,
                "color": (255, 215, 0),  # Gold
                "bold": True,
                "glow": True
            },
            "title": {
                "size_mult": 1.6,
                "color": (255, 220, 100),  # Light gold
                "bold": True,
                "glow": True
            },
            "critical": {
                "size_mult": 1.4,
                "color": (255, 100, 100),  # Red
                "bold": True,
                "glow": True
            },
            "question": {
                "size_mult": 1.3,
                "color": (150, 200, 255),  # Light blue
                "bold": False,
                "glow": False
            },
            "important": {
                "size_mult": 1.2,
                "color": (255, 255, 255),  # White
                "bold": True,
                "glow": False
            },
            "normal": {
                "size_mult": 1.0,
                "color": (230, 230, 230),  # Light gray
                "bold": False,
                "glow": False
            },
            "minor": {
                "size_mult": 0.7,
                "color": (180, 180, 180),  # Gray
                "bold": False,
                "glow": False
            }
        }
    
    def load_enriched_transcript(self, json_path: str) -> Dict[int, List[SemanticWord3D]]:
        """Load enriched transcript and group by scene"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        scenes = {}
        for item in data['phrases']:
            scene_idx = item['appearance_index']
            if scene_idx not in scenes:
                scenes[scene_idx] = []
            
            # Get style for this phrase
            style = self.emphasis_styles.get(
                item['emphasis_type'], 
                self.emphasis_styles["normal"]
            )
            
            # Create words for this phrase
            words = item['text'].split()
            for i, word in enumerate(words):
                word_obj = SemanticWord3D(
                    text=word,
                    start_time=item['start_time'] + (i * 0.05),  # Stagger
                    end_time=item['end_time'],
                    importance=item['importance'],
                    emphasis_type=item['emphasis_type'],
                    appearance_index=scene_idx,
                    layout_priority=item['layout_priority'],
                    font_size_mult=style['size_mult'],
                    color=style['color'],
                    bold=style['bold'],
                    glow=style['glow']
                )
                scenes[scene_idx].append(word_obj)
        
        return scenes
    
    def process_video(self, video_path: str, output_path: str,
                     enriched_json: str, duration: float = 30):
        """Process video with semantic 3D word animations"""
        
        print("Loading enriched transcript...")
        scenes = self.load_enriched_transcript(enriched_json)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {width}x{height} @ {fps} fps")
        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_path = output_path.replace('.mp4', '_temp.mp4')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        # Initialize 3D animation configs for each scene
        scene_animations = {}
        for scene_idx, words in scenes.items():
            # Create text for this scene
            scene_text = ' '.join([w.text for w in words])
            
            # Get dominant style for the scene
            dominant_word = max(words, key=lambda w: w.importance)
            
            # Configure 3D animation
            config = Animation3DConfig(
                text=scene_text,
                position=(width//2, height//2),
                font_size=int(self.base_font_size * dominant_word.font_size_mult),
                color=dominant_word.color,
                bold=dominant_word.bold,
                duration=2.0,  # Rise animation duration
                animation_type='rise',
                rise_height=100,
                start_frame=int(words[0].start_time * fps),
                end_frame=int((words[-1].end_time + self.dissolve_duration) * fps)
            )
            
            # Create animation
            animation = WordRiseSequence3D(config)
            animation.setup(width, height)
            scene_animations[scene_idx] = animation
        
        # Initialize fog dissolve
        fog_effect = CleanFogDissolve(font_size=self.base_font_size)
        
        # Process frames
        total_frames = int(fps * duration)
        frame_count = 0
        
        print("Rendering semantic 3D animation...")
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                # Loop video if needed
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            
            current_time = frame_count / fps
            
            # Find current scene
            current_scene = None
            for scene_idx, words in scenes.items():
                scene_start = words[0].start_time
                scene_end = words[-1].end_time + self.dissolve_duration
                if scene_start <= current_time <= scene_end:
                    current_scene = scene_idx
                    break
            
            # Apply 3D animation for current scene
            if current_scene is not None and current_scene in scene_animations:
                animation = scene_animations[current_scene]
                words = scenes[current_scene]
                
                # Calculate animation progress
                scene_start = words[0].start_time
                scene_end = words[-1].end_time
                
                if current_time < scene_end:
                    # Rising/display phase - use 3D animation
                    frame = animation.process_frame(frame, frame_count)
                    
                    # Add glow effect for important words
                    if words[0].glow:
                        # Simple glow effect
                        kernel = np.ones((5,5), np.float32) / 25
                        blur = cv2.filter2D(frame, -1, kernel)
                        frame = cv2.addWeighted(frame, 0.8, blur, 0.2, 0)
                else:
                    # Dissolve phase - use fog effect
                    dissolve_progress = (current_time - scene_end) / self.dissolve_duration
                    dissolve_progress = min(1.0, dissolve_progress)
                    
                    # Get text and position from animation config
                    text = animation.config.text
                    pos = animation.config.position
                    
                    # Apply fog dissolve
                    frame = fog_effect.apply_clean_fog(
                        text, frame, pos, dissolve_progress
                    )
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % (fps * 3) == 0:
                print(f"  Progress: {frame_count * 100 // total_frames}%")
        
        cap.release()
        out.release()
        
        # Add audio
        print("Adding audio and converting to H.264...")
        import subprocess
        result = subprocess.run([
            'ffmpeg', '-y',
            '-i', temp_path,
            '-ss', '0', '-t', str(duration),
            '-i', video_path,
            '-map', '0:v', '-map', '1:a',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'copy',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            output_path
        ], capture_output=True)
        
        if result.returncode == 0:
            os.remove(temp_path)
            print(f"✅ Semantic 3D video created: {output_path}")
        else:
            print(f"FFmpeg error: {result.stderr.decode()}")


def main():
    """Test semantic 3D pipeline"""
    print("SEMANTIC 3D WORD ANIMATION PIPELINE")
    print("=" * 50)
    print("Using WordRiseSequence3D with semantic emphasis\n")
    
    # Regenerate enriched transcript
    print("Regenerating enriched transcript...")
    enricher = TranscriptEnricher()
    transcript_path = "uploads/assets/videos/ai_math1/transcript_elevenlabs_scribe.json"
    enriched_path = "uploads/assets/videos/ai_math1/transcript_enriched.json"
    
    if os.path.exists(transcript_path):
        enriched = enricher.process_transcript(transcript_path, enriched_path)
        print(f"Created {len(enriched)} phrases\n")
    
    # Create pipeline
    pipeline = Semantic3DWordPipeline()
    
    # Process video
    video_path = "uploads/assets/videos/ai_math1/ai_math1_final.mp4"
    output_path = "outputs/ai_math1_semantic_3d_30s_h264.mp4"
    
    pipeline.process_video(video_path, output_path, enriched_path, duration=30)
    
    print("\n✨ Success! Semantic 3D video ready:", output_path)
    print("\nFeatures:")
    print("  • 3D word rise animation (WordRiseSequence3D)")
    print("  • Semantic emphasis with size/color/glow")
    print("  • Proper fog dissolve transitions")
    print("  • Scene-based text grouping")
    print("  • Based on proven 3D animation system")


if __name__ == "__main__":
    main()