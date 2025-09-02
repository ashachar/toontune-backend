# Fix for Duplicate Text Appearance Issue

## Problem
Text with whole-phrase entrance effects (like `fade_slide_bottom`) and word timings would appear twice:
1. First: The entire phrase would slide/fade in during entrance
2. Then: Words would disappear and reappear based on their individual timestamps

Example: "AI created new math" would:
- Slide in from bottom as a complete phrase
- Then disappear
- Then each word would reappear at its specific timestamp

## Root Cause
The rendering logic had a flaw when handling phrases with BOTH:
- Word timings (individual start/end times for each word)
- Non-word-by-word entrance effects (like `fade_slide_top`, `fade_slide_bottom`)

The code would:
1. During ENTRANCE phase: Show ALL words with the entrance animation
2. During STEADY phase: Apply word timing gates, hiding words whose time hadn't come yet
3. This caused words to disappear after entrance and reappear at their timestamps

## The Fix
Implemented three distinct rendering paths:

### 1. Word-by-Word Effect Path
- **When**: `entrance_effect == FADE_WORD_BY_WORD` AND has word timings
- **Behavior**: Progressive slot-based reveal of individual words
- **Code**: Lines 313-388

### 2. Whole-Phrase Effect with Word Timings Path  
- **When**: Has word timings but entrance effect is NOT word-by-word
- **Behavior**: Render only visible words based on timestamps, but as a single unit
- **Code**: Lines 389-422
- **Key logic**: Build visible text from words where `current_time >= word_start`

### 3. Simple Whole-Phrase Path
- **When**: No word timings at all
- **Behavior**: Render entire phrase with entrance/disappearance animations
- **Code**: Lines 423-445

## Key Implementation Details

```python
# Decision logic
use_word_by_word_rendering = (
    "word_timings" in phrase 
    and phrase["word_timings"]
    and entrance_effect == EntranceEffect.FADE_WORD_BY_WORD
)

if use_word_by_word_rendering:
    # Path 1: Individual word rendering with progressive slots
    # Each word appears in its time slot during entrance
elif "word_timings" in phrase and phrase["word_timings"]:
    # Path 2: Whole-phrase effect respecting word timings
    # Only render words that have started, but render them together
    visible_words = [w for w, t in zip(words, timings) if current_time >= t["start"]]
    visible_text = " ".join(visible_words)
    # Apply whole-phrase entrance effect to visible portion
else:
    # Path 3: Simple whole phrase, no timing restrictions
    # Render entire text with animations
```

## Result
- Text appears exactly once with its assigned entrance effect
- Word timings are always respected
- No duplicate appearances or disappearing/reappearing text
- Whole-phrase effects (slide/fade) work correctly with partial text visibility

## Testing
Verified with "AI created new math" which now:
- Uses `fade_slide_bottom` effect
- Respects word timings (words appear as they should)
- No duplicate rendering
- Smooth entrance animation applied to visible portion only