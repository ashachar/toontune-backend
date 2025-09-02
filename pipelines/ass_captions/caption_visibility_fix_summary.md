# Caption Visibility Fix Summary

## Problem Identified
- Captions were not visible in the full video output
- ALL 1,448 phrases were being placed behind the person mask
- Zero phrases were rendered in front, making them invisible

## Root Cause
- The visibility threshold was set to 90% (0.9)
- The person in the video covers 40-50% of the text area at the bottom
- With 90% threshold, ANY text with <90% visibility goes behind
- Since the person covers ~45% of bottom area, visibility was always ~55%
- Result: ALL text went behind the person and was mostly hidden

## Solution Applied
- Lowered visibility threshold from 0.9 (90%) to 0.4 (40%)
- Now text only goes behind if <40% would be visible
- With typical 55% visibility, text now stays in FRONT

## Verification
- Test shows bottom text (54.7% visible) now stays FRONT ✅
- Top text (75.7% visible) also stays FRONT ✅
- Processing full video with new threshold...

## Files Modified
- `sam2_head_aware_sandwich.py`: Line 2254 changed threshold to 0.4
