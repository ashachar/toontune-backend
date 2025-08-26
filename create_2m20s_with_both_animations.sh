#!/bin/bash

echo "========================================================================"
echo "🎬 CREATING HELLO WORLD WITH BOTH ANIMATIONS ON 2:20-2:24"
echo "========================================================================"

# Step 1: Extract the 2:20-2:24 segment
echo ""
echo "📹 Step 1: Extracting 2:20-2:24 segment..."
ffmpeg -i uploads/assets/videos/ai_math1.mp4 -ss 140 -t 4 \
    -c:v copy -c:a copy outputs/ai_math1_2m20s_segment.mp4 -y 2>/dev/null

echo "✅ Segment extracted"

# Step 2: Apply both animations using the existing pipeline
echo ""
echo "🎨 Step 2: Applying motion + dissolve animations..."
python utils/animations/apply_3d_text_animation.py \
    outputs/ai_math1_2m20s_segment.mp4 \
    --text "Hello World" \
    --position center \
    --font-size 72 \
    --motion-duration 0.8 \
    --safety-hold 0.5 \
    --dissolve-duration 2.5 \
    --is-behind \
    --supersample 8 \
    --output outputs/hello_world_2m20s_both_animations.mp4

echo ""
echo "========================================================================"
echo "✅ COMPLETE!"
echo "========================================================================"
echo ""
echo "📊 Animation timeline:"
echo "  • 0.0-0.8s: 3D text MOTION (emergence and movement)"
echo "  • 0.8-1.3s: Safety HOLD (text stable)"
echo "  • 1.3-3.8s: Letter DISSOLVE (gradual disappearance)"
echo ""
echo "🎥 Output: outputs/hello_world_2m20s_both_animations.mp4"
echo ""
echo "Opening video..."
open outputs/hello_world_2m20s_both_animations.mp4
echo "========================================================================"