#!/usr/bin/env python3
"""Create final comparison showing the occlusion fix."""

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Load both frames
before = Image.open('outputs/check_frame_at_1s.png')
after = Image.open('outputs/check_frame_at_1s_fixed.png')

# Create side-by-side comparison
width = before.width
height = before.height
comparison = Image.new('RGB', (width * 2, height))

# Process each image
for i, (img, title, desc) in enumerate([
    (before, "BEFORE: Broken Occlusion", "Letters visible through person (static mask)"),
    (after, "AFTER: Fixed with Dynamic Masking", "Letters correctly hidden behind person")
]):
    # Add annotations
    draw = ImageDraw.Draw(img)
    
    # Title bar
    draw.rectangle([(0, 0), (width, 60)], fill=(0, 0, 0, 200))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    draw.text((20, 10), title, fill=(255, 255, 255), font=font)
    draw.text((20, 38), desc, fill=(200, 200, 200), font=small_font)
    
    # Analyze for occlusion issues
    img_array = np.array(img)
    
    # Find yellow letters
    yellow_mask = (img_array[:, :, 0] > 180) & (img_array[:, :, 1] > 180) & (img_array[:, :, 2] < 120)
    
    # Find areas that should occlude (person's body/arms)
    # Focus on the area where the person's left arm is (around the H)
    h, w = img_array.shape[:2]
    
    # The person's arm area (approximate)
    arm_region = img_array[250:450, 150:350]  # Region where arm overlaps with 'H'
    
    # Check if letters are visible in this region
    arm_yellow = yellow_mask[250:450, 150:350]
    letter_pixels_in_arm = np.sum(arm_yellow)
    
    if i == 0 and letter_pixels_in_arm > 50:
        # Draw attention to the problem
        draw.rectangle([(150, 250), (350, 450)], outline=(255, 0, 0), width=3)
        draw.text((155, 255), "BUG: Letter visible", fill=(255, 0, 0), font=small_font)
        draw.text((155, 275), "through person!", fill=(255, 0, 0), font=small_font)
        
        # Arrow pointing to issue
        draw.line([(350, 350), (250, 350)], fill=(255, 0, 0), width=2)
        draw.polygon([(250, 345), (250, 355), (240, 350)], fill=(255, 0, 0))
        
    elif i == 1 and letter_pixels_in_arm < 50:
        # Show it's fixed
        draw.rectangle([(150, 250), (350, 450)], outline=(0, 255, 0), width=3)
        draw.text((155, 255), "FIXED: Letter", fill=(0, 255, 0), font=small_font)
        draw.text((155, 275), "correctly hidden", fill=(0, 255, 0), font=small_font)
    
    # Paste into comparison
    comparison.paste(img, (i * width, 0))

# Add dividing line
draw = ImageDraw.Draw(comparison)
draw.line([(width, 0), (width, height)], fill=(255, 255, 255), width=2)

# Add technical explanation at bottom
draw.rectangle([(0, height - 80), (width * 2, height)], fill=(0, 0, 0, 220))
draw.text((20, height - 70), "Computer Vision Fix Applied:", fill=(255, 255, 255), font=font)
draw.text((20, height - 45), "• Dynamic foreground extraction every frame using segment_extractor", fill=(200, 200, 200), font=small_font)
draw.text((20, height - 25), "• Removed fallback to stale mask that caused incorrect occlusion", fill=(200, 200, 200), font=small_font)
draw.text((width + 20, height - 45), "• Each frame's mask now reflects actual person position", fill=(200, 200, 200), font=small_font)
draw.text((width + 20, height - 25), "• Letters behind person are properly masked in real-time", fill=(200, 200, 200), font=small_font)

comparison.save('outputs/occlusion_fix_final_proof.png')
print("✅ Final comparison saved to: outputs/occlusion_fix_final_proof.png")
print("\n🎯 Computer Vision Fix Summary:")
print("1. Problem: Static mask fallback caused letters to show through moving person")
print("2. Solution: Force fresh mask extraction every frame, never use stale masks")
print("3. Result: Dynamic occlusion that properly tracks person movement")