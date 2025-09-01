#!/usr/bin/env python3
"""
Simple eraser wipe animation using FFmpeg filters
"""

import subprocess
import os
import numpy as np
import cv2


def create_eraser_wipe_with_ffmpeg(input_video: str, eraser_image: str, output_video: str, 
                                   wipe_start: float = 2.7, wipe_duration: float = 0.6):
    """
    Create eraser wipe animation using FFmpeg filters
    
    The eraser moves in a circular pattern around the character,
    and reveals the original video underneath.
    """
    
    # First, extract a frame to get character position
    cap = cv2.VideoCapture(input_video)
    cap.set(cv2.CAP_PROP_POS_MSEC, wipe_start * 1000)
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    cap.release()
    
    # Find character center (simple approach - find non-green pixels)
    # Convert to HSV to detect green screen
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define green range
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    
    # Create mask for non-green (character) pixels
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    character_mask = cv2.bitwise_not(green_mask)
    
    # Find character bounding box
    coords = np.column_stack(np.where(character_mask > 0))
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Character center and radius for circular motion
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        radius_x = (x_max - x_min) // 2 + 50  # Add margin
        radius_y = (y_max - y_min) // 2 + 50
    else:
        # Default to center of frame
        center_x = width // 2
        center_y = height // 2
        radius_x = 200
        radius_y = 250
    
    # Create path points for eraser motion (circular/elliptical path)
    num_points = 20
    path_points = []
    for i in range(num_points + 1):  # +1 to complete the circle
        angle = 2 * np.pi * i / num_points
        x = int(center_x + radius_x * np.cos(angle))
        y = int(center_y + radius_y * np.sin(angle) * 0.8)  # Flatten vertically
        path_points.append((x, y))
    
    # Build FFmpeg filter for animated eraser with reveal effect
    # We'll use multiple overlay filters with different timing
    
    filter_parts = []
    
    # For each point in the path, add an overlay at specific time
    for i, (x, y) in enumerate(path_points[:-1]):  # Skip last point (same as first)
        t_start = wipe_start + (i * wipe_duration / len(path_points))
        t_end = wipe_start + ((i + 1) * wipe_duration / len(path_points))
        
        # Calculate eraser position (offset for eraser tip)
        eraser_x = x - 150  # Half of scaled eraser width
        eraser_y = y - 100  # Offset for eraser tip
        
        # Add overlay for this time segment
        if i == 0:
            filter_parts.append(f"[0:v][1:v]overlay=x='{eraser_x}':y='{eraser_y}':enable='between(t,{t_start},{t_end})'[v{i}]")
        else:
            filter_parts.append(f"[v{i-1}][1:v]overlay=x='{eraser_x}':y='{eraser_y}':enable='between(t,{t_start},{t_end})'[v{i}]")
    
    # Join all filter parts
    filter_complex = ";".join(filter_parts)
    
    # Simple approach: just animate the eraser moving
    # For actual pixel removal, we'd need a more complex approach
    
    # Create a simple moving eraser overlay
    # Using expressions to animate position over time
    t_start = wipe_start
    t_end = wipe_start + wipe_duration
    
    # Create expression for circular motion
    # x(t) = center_x + radius * cos(2*pi*(t-t_start)/duration)
    # y(t) = center_y + radius * sin(2*pi*(t-t_start)/duration)
    
    x_expr = f"{center_x}+{radius_x}*cos(2*PI*(t-{t_start})/{wipe_duration})-150"
    y_expr = f"{center_y}+{radius_y*0.8}*sin(2*PI*(t-{t_start})/{wipe_duration})-100"
    
    # Build FFmpeg command with moving eraser
    cmd = [
        'ffmpeg', '-y',
        '-i', input_video,
        '-i', eraser_image,
        '-filter_complex',
        f"[1:v]scale=300:-1[eraser];"
        f"[0:v][eraser]overlay="
        f"x='{x_expr}':y='{y_expr}':"
        f"enable='between(t,{t_start},{t_end})'",
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    print(f"Creating eraser animation...")
    print(f"Character center: ({center_x}, {center_y})")
    print(f"Motion radius: ({radius_x}, {radius_y})")
    print(f"Animation from {t_start}s to {t_end}s")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    print(f"Eraser animation created: {output_video}")
    return True


def create_eraser_wipe_with_mask(character_video: str, original_video: str, 
                                 eraser_image: str, output_video: str,
                                 wipe_start: float = 2.7, wipe_duration: float = 0.6):
    """
    Create eraser wipe that actually reveals original video underneath
    """
    
    # This is complex in FFmpeg, so let's create a mask video first
    # that gradually reveals the original
    
    # For now, simplified version with blend transition
    t_start = wipe_start
    t_end = wipe_start + wipe_duration
    
    cmd = [
        'ffmpeg', '-y',
        '-i', character_video,  # Character video (to be erased)
        '-i', original_video,   # Original video (to be revealed)
        '-i', eraser_image,     # Eraser image
        '-filter_complex',
        # Scale eraser
        f"[2:v]scale=300:-1[eraser];"
        # Create moving eraser overlay
        f"[0:v][eraser]overlay="
        f"x='640+200*cos(2*PI*(t-{t_start})/{wipe_duration})-150':"
        f"y='360+150*sin(2*PI*(t-{t_start})/{wipe_duration})-100':"
        f"enable='between(t,{t_start},{t_end})'[with_eraser];"
        # Blend to original during wipe
        f"[with_eraser][1:v]blend=all_expr='if(gte(T,{t_start})*lte(T,{t_end}),"
        f"A*(1-(T-{t_start})/{wipe_duration})+B*(T-{t_start})/{wipe_duration},A)'",
        '-c:v', 'libx264',
        '-preset', 'fast', 
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    print(f"Creating eraser wipe with reveal...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
        
    print(f"Eraser wipe created: {output_video}")
    return True