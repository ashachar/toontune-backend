#!/usr/bin/env python3
"""Test first 30 seconds with fixed visibility threshold"""

import sys
sys.path.insert(0, '.')

from sam2_head_aware_sandwich import apply_sam2_sandwich_compositing

# Process just first 30 seconds
apply_sam2_sandwich_compositing(
    "../../uploads/assets/videos/ai_math1.mp4",
    "../../uploads/assets/videos/ai_math1/ai_math1_rvm_mask.mp4", 
    "../../uploads/assets/videos/ai_math1/transcript_enriched.json",
    "../../outputs/ai_math1_30s_test.mp4",
    visibility_threshold=0.4,  # New threshold
    max_seconds=30  # Process only first 30 seconds
)

print("\n✅ Test video created: ../../outputs/ai_math1_30s_test_h264.mp4")
