# Baseline Alignment Fix Summary

## Problem
Words in the same phrase (especially behind-text like "surprised if") were not baseline-aligned. Each word appeared at different vertical positions because we were calculating Y positions based on individual word heights.

## Root Cause
The code was calculating baseline position using `max_height // 2` and then subtracting each word's individual height. This caused words with different heights (especially with/without descenders) to appear misaligned.

## Solution
**Measure the full phrase as a single text string and use the same Y position for all words in that phrase.**

### Key Changes in `word_factory.py`:

#### Before:
```python
# Calculate baseline from max word height
max_height = max(h for w, h in word_measurements)
baseline_y = center_y + (max_height // 2)

# Each word got different Y based on its height
top_y = baseline_y - height  # Different for each word!
```

#### After:
```python
# Measure the complete phrase
full_phrase_bbox = draw.textbbox((0, 0), text, font=font)
phrase_height = full_phrase_bbox[3] - full_phrase_bbox[1]
phrase_top_y = center_y - (phrase_height // 2)

# All words use the SAME Y position
top_y = phrase_top_y  # Same for all words in phrase!
```

## Why This Works
When the font renderer draws text with `anchor="lt"` (left-top), it automatically handles baseline alignment. By giving all words in a phrase the same top Y position, they naturally align on the same baseline, just as they would if rendered as a complete phrase.

## Result
✅ All words in a phrase now share the same baseline
✅ Works consistently for both front and behind text
✅ Text appears properly aligned as in professional typography

## Test Output
- `outputs/ai_math1_would_bug_fixed.mp4` - Video with properly aligned text
- `outputs/ai_math1_would_bug_fixed_debug.mp4` - Debug version showing bounding boxes

The fix ensures that phrases like "surprised if" have all words sitting on the same invisible baseline, regardless of individual character heights or descenders.