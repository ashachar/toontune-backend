#!/usr/bin/env python3
"""
VFX-based photorealistic burning text animation.
Uses real fire/smoke overlay techniques inspired by ProductionCrate assets.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import random

@dataclass
class BurnPhase:
    """Represents a phase in the burning sequence"""
    name: str
    start_time: float
    duration: float
    intensity: float

class VFXBurn:
    """Photorealistic burning using VFX overlay techniques"""
    
    def __init__(
        self,
        text: str,
        font_size: int = 120,
        resolution: Tuple[int, int] = (1280, 720),
        duration: float = 4.0,
        fps: int = 30
    ):
        self.text = text
        self.font_size = font_size
        self.width, self.height = resolution
        self.duration = duration
        self.fps = fps
        self.total_frames = int(duration * fps)
        
        # Burn phases based on real fire behavior
        self.phases = [
            BurnPhase("ignition", 0.0, 0.3, 0.3),
            BurnPhase("spreading", 0.2, 0.6, 0.7),
            BurnPhase("peak_burn", 0.5, 0.8, 1.0),
            BurnPhase("charring", 1.0, 1.0, 0.6),
            BurnPhase("smoldering", 1.5, 1.5, 0.3),
            BurnPhase("ashing", 2.0, 2.0, 0.1)
        ]
        
        # Pre-render text
        self.text_mask = self._create_text_mask()
        self.letter_masks = self._separate_letters()
        
    def _create_text_mask(self) -> np.ndarray:
        """Create base text mask"""
        img = np.zeros((self.height, self.width), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_DUPLEX
        
        # Get text size and center it
        (text_w, text_h), _ = cv2.getTextSize(self.text, font, self.font_size/30, 3)
        x = (self.width - text_w) // 2
        y = (self.height + text_h) // 2
        
        cv2.putText(img, self.text, (x, y), font, self.font_size/30, 255, 3)
        return img
    
    def _separate_letters(self) -> List[np.ndarray]:
        """Separate text into individual letter masks"""
        contours, _ = cv2.findContours(self.text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours left to right
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        letter_masks = []
        for contour in contours:
            mask = np.zeros_like(self.text_mask)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            letter_masks.append(mask)
            
        return letter_masks
    
    def _generate_fire_glow(self, mask: np.ndarray, intensity: float, color_temp: float) -> np.ndarray:
        """Generate fire glow effect"""
        # Create color gradient based on fire temperature
        glow = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        # Dilate mask for glow spread
        kernel_size = int(15 * intensity)
        if kernel_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated = cv2.dilate(mask, kernel)
            
            # Apply Gaussian blur for soft glow
            glow_mask = cv2.GaussianBlur(dilated.astype(np.float32), (31, 31), 10)
            glow_mask /= 255.0
            
            # Fire color gradient (blue -> white -> yellow -> orange -> red)
            if color_temp > 0.8:  # Very hot - blue/white
                glow[:,:,0] = glow_mask * 255 * intensity  # Blue
                glow[:,:,1] = glow_mask * 240 * intensity  # Green
                glow[:,:,2] = glow_mask * 255 * intensity  # Red
            elif color_temp > 0.6:  # Hot - yellow/white
                glow[:,:,0] = glow_mask * 100 * intensity  # Blue
                glow[:,:,1] = glow_mask * 200 * intensity  # Green
                glow[:,:,2] = glow_mask * 255 * intensity  # Red
            elif color_temp > 0.3:  # Medium - orange
                glow[:,:,0] = glow_mask * 20 * intensity   # Blue
                glow[:,:,1] = glow_mask * 100 * intensity  # Green
                glow[:,:,2] = glow_mask * 255 * intensity  # Red
            else:  # Cool - deep red
                glow[:,:,0] = glow_mask * 0 * intensity    # Blue
                glow[:,:,1] = glow_mask * 30 * intensity   # Green
                glow[:,:,2] = glow_mask * 180 * intensity  # Red
                
        return glow.astype(np.uint8)
    
    def _add_smoke_particles(self, frame: np.ndarray, source_mask: np.ndarray, 
                            intensity: float, frame_num: int) -> np.ndarray:
        """Add volumetric smoke effect"""
        if intensity < 0.1:
            return frame
            
        smoke_frame = frame.copy()
        
        # Find smoke emission points (top edge of burning area)
        contours, _ = cv2.findContours(source_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Generate smoke particles from top edge
            num_particles = int(10 * intensity)
            for _ in range(num_particles):
                # Random position along top edge
                px = x + random.randint(0, w)
                py = y - random.randint(0, 20)
                
                # Smoke rises with turbulence
                drift_x = random.randint(-5, 5)
                rise_y = -random.randint(5, 15) * (frame_num % 10) / 10
                
                smoke_x = int(px + drift_x)
                smoke_y = int(py + rise_y)
                
                if 0 <= smoke_x < self.width and 0 <= smoke_y < self.height:
                    # Draw smoke particle with transparency
                    radius = random.randint(5, 15)
                    color = int(100 * intensity)  # Grayish smoke
                    overlay = smoke_frame.copy()
                    cv2.circle(overlay, (smoke_x, smoke_y), radius, 
                             (color, color, color), -1)
                    smoke_frame = cv2.addWeighted(smoke_frame, 0.8, overlay, 0.2, 0)
                    
        return smoke_frame
    
    def _create_char_texture(self, mask: np.ndarray, char_amount: float) -> np.ndarray:
        """Create charred/burned texture"""
        char_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if char_amount > 0:
            # Create char pattern with noise
            noise = np.random.randint(0, 50, (self.height, self.width))
            
            # Apply to mask area
            char_color = int(30 + 50 * (1 - char_amount))  # Darker as more charred
            
            for c in range(3):
                char_img[:,:,c] = (mask > 0) * (char_color + noise)
                
            # Add cracks/texture
            if char_amount > 0.5:
                # Add some crack lines
                num_cracks = int(5 * char_amount)
                for _ in range(num_cracks):
                    pt1 = (random.randint(0, self.width-1), random.randint(0, self.height-1))
                    pt2 = (pt1[0] + random.randint(-30, 30), pt1[1] + random.randint(-30, 30))
                    # Ensure pt1 is within bounds
                    if 0 <= pt1[1] < self.height and 0 <= pt1[0] < self.width:
                        if mask[pt1[1], pt1[0]] > 0:
                            cv2.line(char_img, pt1, pt2, (10, 10, 10), 1)
                        
        return char_img
    
    def _add_embers(self, frame: np.ndarray, mask: np.ndarray, 
                    intensity: float, frame_num: int) -> np.ndarray:
        """Add glowing ember particles"""
        if intensity < 0.2:
            return frame
            
        ember_frame = frame.copy()
        
        # Find edges for ember placement
        edges = cv2.Canny(mask, 100, 200)
        edge_points = np.column_stack(np.where(edges > 0))
        
        if len(edge_points) > 0:
            num_embers = min(int(20 * intensity), len(edge_points))
            selected_points = edge_points[np.random.choice(len(edge_points), num_embers, replace=False)]
            
            for point in selected_points:
                y, x = point
                
                # Ember glow with pulsing
                pulse = 0.5 + 0.5 * np.sin(frame_num * 0.3 + random.random() * np.pi)
                ember_intensity = intensity * pulse
                
                # Draw ember
                ember_color = (
                    int(50 * ember_intensity),   # Blue (minimal)
                    int(100 * ember_intensity),  # Green
                    int(255 * ember_intensity)   # Red (dominant)
                )
                
                cv2.circle(ember_frame, (x, y), 2, ember_color, -1)
                
                # Add glow around ember
                overlay = ember_frame.copy()
                cv2.circle(overlay, (x, y), 6, ember_color, -1)
                ember_frame = cv2.addWeighted(ember_frame, 0.7, overlay, 0.3, 0)
                
        return ember_frame
    
    def generate_frame(self, frame_num: int) -> np.ndarray:
        """Generate a single frame of burning animation"""
        # Create base frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (20, 20, 30)  # Dark background
        
        # Calculate animation progress
        t = frame_num / self.total_frames
        
        # Process each letter with staggered timing
        for i, letter_mask in enumerate(self.letter_masks):
            # Stagger the burn start for each letter
            letter_delay = i * 0.15
            letter_t = max(0, min(1, (t - letter_delay) / (1 - letter_delay)))
            
            # Determine current phase
            current_intensity = 0
            color_temp = 0
            char_amount = 0
            
            for phase in self.phases:
                phase_t = (letter_t * self.duration - phase.start_time) / phase.duration
                if 0 <= phase_t <= 1:
                    if phase.name == "ignition":
                        current_intensity = phase.intensity * phase_t
                        color_temp = 0.9
                    elif phase.name == "spreading":
                        current_intensity = phase.intensity
                        color_temp = 0.7 - 0.3 * phase_t
                    elif phase.name == "peak_burn":
                        current_intensity = phase.intensity
                        color_temp = 0.4
                    elif phase.name == "charring":
                        current_intensity = phase.intensity * (1 - phase_t)
                        color_temp = 0.2
                        char_amount = phase_t
                    elif phase.name == "smoldering":
                        current_intensity = phase.intensity * (1 - phase_t)
                        char_amount = 0.8 + 0.2 * phase_t
                    elif phase.name == "ashing":
                        char_amount = 1.0 * (1 - phase_t)
            
            # Apply effects in layers
            if letter_t < 0.9:  # Still visible
                # Add base text (fading as it burns)
                text_opacity = max(0, 1 - char_amount)
                if text_opacity > 0:
                    text_color = (200, 200, 200)
                    text_img = cv2.cvtColor(letter_mask, cv2.COLOR_GRAY2BGR)
                    text_img = (text_img > 0) * text_color
                    frame = cv2.addWeighted(frame, 1, text_img.astype(np.uint8), 
                                          text_opacity, 0)
                
                # Add char texture
                if char_amount > 0:
                    char_texture = self._create_char_texture(letter_mask, char_amount)
                    frame = cv2.addWeighted(frame, 1, char_texture, char_amount, 0)
                
                # Add fire glow
                if current_intensity > 0:
                    glow = self._generate_fire_glow(letter_mask, current_intensity, color_temp)
                    frame = cv2.add(frame, glow)
                
                # Add embers
                frame = self._add_embers(frame, letter_mask, current_intensity, frame_num)
                
                # Add smoke
                frame = self._add_smoke_particles(frame, letter_mask, current_intensity, frame_num)
        
        return frame

def create_vfx_burn_demo():
    """Create a demo of the VFX burn effect"""
    print("Creating VFX-based photorealistic burn animation...")
    
    burn = VFXBurn(
        text="FIRE",
        font_size=150,
        resolution=(1280, 720),
        duration=4.0,
        fps=30
    )
    
    # Create output video
    output_path = "outputs/vfx_burn_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, burn.fps, 
                           (burn.width, burn.height))
    
    print(f"Generating {burn.total_frames} frames...")
    for frame_num in range(burn.total_frames):
        frame = burn.generate_frame(frame_num)
        writer.write(frame)
        
        if frame_num % 30 == 0:
            print(f"  Frame {frame_num}/{burn.total_frames}")
    
    writer.release()
    
    # Convert to H.264
    h264_path = "outputs/vfx_burn_demo_h264.mp4"
    import os
    os.system(f"ffmpeg -i {output_path} -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart -y {h264_path} 2>/dev/null")
    
    print(f"✅ VFX burn demo saved to: {h264_path}")
    return h264_path

if __name__ == "__main__":
    create_vfx_burn_demo()