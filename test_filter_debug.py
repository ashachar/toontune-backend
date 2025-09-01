#!/usr/bin/env python3
"""Debug the filter complex construction"""

import math

# Parameters from smooth_eraser_wipe.py
width = 1280
height = 720
center_x = width // 2 - 20  # Shift left to align with character
center_y = height // 2 + 10
radius_x = 200
radius_y = 150

scale_factor = 0.70
pivot_x_ratio = 0.1953
pivot_y_ratio = 0.62
tip_ratio = 0.12
top_margin = 20
amplitude = radius_y * 0.8
bottom_safety = 2
contact_x_ratio = 0.08  # Left edge of eraser

# Calculate offsets
scaled_width = int(768 * scale_factor)
scaled_height = int(1344 * scale_factor)
contact_dx = scaled_width * (contact_x_ratio - pivot_x_ratio)

# Brush parameters
brush_size = 150
brush_feather = 6.0
sample_fps = 60
wipe_start = 0
wipe_duration = 0.6
num_samples = max(1, int(wipe_duration * sample_fps))

print(f"Width: {width}, Height: {height}")
print(f"Center: ({center_x}, {center_y})")
print(f"Scaled eraser: {scaled_width}x{scaled_height}")
print(f"Contact dx: {contact_dx}")
print(f"Num samples: {num_samples}")
print()

# Build filter complex
filter_parts = []

# 1. Prepare input streams
filter_parts.append("[0:v]format=rgba,split=2[char][char_ref];")
filter_parts.append("[1:v]format=rgba[orig];")
filter_parts.append(f"[2:v]format=rgba,scale=iw*{scale_factor}:ih*{scale_factor}[eraser];")

# 2. Create empty full-frame mask
filter_parts.append("color=c=black@0.0:s=16x16:rate=60,format=rgba[mask_seed];")
filter_parts.append("[mask_seed][char_ref]scale2ref[mask0][char_sized];")

# 3. Create soft circular brush
r = brush_size // 2 - 1
geq_expr = f"if(lte((X-{r})*(X-{r})+(Y-{r})*(Y-{r}),{r*r}),255,0)"
filter_parts.append(
    f"color=c=black:s={brush_size}x{brush_size}:rate={sample_fps},format=gray,"
    f"geq=lum='{geq_expr}',gblur=sigma={brush_feather}[brush_alpha];"
)
filter_parts.append(
    f"color=c=white@1.0:s={brush_size}x{brush_size}:rate={sample_fps},format=rgba[brush_rgb];"
)
filter_parts.append("[brush_rgb][brush_alpha]alphamerge[brush];")

# 4. Accumulate mask by stamping brush along path
current_mask = "mask0"
for i in range(num_samples):
    progress = i / (num_samples - 1) if num_samples > 1 else 0
    t = wipe_start + progress * wipe_duration
    angle = 2 * math.pi * progress
    
    # Calculate brush position (contact point)
    x_contact = int(center_x + radius_x * math.cos(angle) + contact_dx)
    
    # Y position with proper calculations
    sin_val = math.sin(angle)
    y_top_raw = center_y + amplitude * sin_val - scaled_height * pivot_y_ratio + \
               (top_margin - (center_y - amplitude) + scaled_height * (pivot_y_ratio - tip_ratio))
    y_top = max(y_top_raw, height - scaled_height + bottom_safety)
    y_tip = int(y_top + scaled_height * tip_ratio)
    
    # Convert to brush top-left position
    brush_x = x_contact - brush_size // 2
    brush_y = y_tip - brush_size // 2
    
    # Stamp brush with persistence
    filter_parts.append(
        f"[{current_mask}][brush]overlay="
        f"x={brush_x}:y={brush_y}:eval=frame:shortest=0:eof_action=pass:"
        f"enable='gte(t,{t:.6f})'[mask{i+1}];"
    )
    current_mask = f"mask{i+1}"

# 5. Extract mask alpha for maskedmerge
filter_parts.append(f"[{current_mask}]alphaextract,format=gray[revealmask];")

# 6. Use mask to reveal original over character
filter_parts.append("[orig][char][revealmask]maskedmerge[reveal];")

# 7. Overlay moving eraser on top (visual element)
y_offset = f"{top_margin} - ({center_y} - {amplitude}) + overlay_h*({pivot_y_ratio} - {tip_ratio})"
y_raw = f"{center_y}+{amplitude}*sin(2*PI*(t-{wipe_start})/{wipe_duration})-overlay_h*{pivot_y_ratio}+({y_offset})"
min_y = f"main_h - overlay_h + {bottom_safety}"

eraser_motion = (
    f"overlay="
    f"x='{center_x}+{radius_x}*cos(2*PI*(t-{wipe_start})/{wipe_duration})-overlay_w*{pivot_x_ratio}':"
    f"y='max({y_raw},{min_y})':"
    f"eval=frame:shortest=0:eof_action=pass:"
    f"enable='between(t,{wipe_start},{wipe_start + wipe_duration + 0.02})'"
)

filter_parts.append(f"[reveal][eraser]{eraser_motion}[with_eraser];")

# 8. Final format - null the unused char_sized output
filter_parts.append("[with_eraser]format=yuv420p[outv];")
filter_parts.append("[char_sized]nullsink")

filter_complex = "".join(filter_parts)

print("Filter complex length:", len(filter_complex))
print("\nFirst 500 chars:")
print(filter_complex[:500])
print("\nLast 500 chars:")
print(filter_complex[-500:])

# Check for any obvious issues
if filter_complex.count('[') != filter_complex.count(']'):
    print("\n❌ ERROR: Unmatched brackets!")
if filter_complex.count('(') != filter_complex.count(')'):
    print("\n❌ ERROR: Unmatched parentheses!")

# Count the number of filters
num_filters = filter_complex.count(';')
print(f"\nNumber of filter statements: {num_filters}")