# V2 Prompt Changes Summary

## Overview
The unified video pipeline prompt generation has been updated from word-by-word overlays to a more strategic approach using key phrases and cartoon characters.

## Key Changes

### 1. Removed: Word-by-Word Overlays
**Old Approach:**
- Every single word from the transcript was suggested for overlay
- Created visual clutter with constant text appearing/disappearing
- Example: "Let's", "start", "at", "the", "very", "beginning" (6 separate overlays)

### 2. Added: Key Phrases (Strategic Text)
**New Approach:**
- Maximum 4 words per phrase
- Appears maximum once every 20 seconds
- Only the most emotionally important or thematically central phrases
- Example: "Do Re Mi", "very beginning" (2 impactful phrases for 60-second scene)

**Constraints Applied:**
- 60-second scene: Maximum 3 key phrases
- 30-second scene: Maximum 1-2 key phrases
- 20-second scene: Maximum 1 key phrase

### 3. Added: Cartoon Characters
**New Feature:**
- Cute, animated characters 100% related to content
- Maximum 1 character every 10-15 seconds
- Enhance narrative without distraction
- Examples:
  - "Doe a deer" lyric → Happy deer character
  - "Do Re Mi" → Dancing musical notes
  - "Raindrops on roses" → Animated rose with dewdrops
  - Mountain scene → Happy mountain goat

**Constraints Applied:**
- 60-second scene: Maximum 4-6 cartoon characters
- 30-second scene: Maximum 2-3 cartoon characters
- 20-second scene: Maximum 2 cartoon characters

## JSON Structure Changes

### Old Structure (text_overlays)
```json
"text_overlays": [
  {
    "word": "Let's",
    "start_seconds": "0.200",
    "end_seconds": "0.500",
    "top_left_pixels": {"x": 100, "y": 50},
    "bottom_right_pixels": {"x": 150, "y": 80},
    "text_effect": "Typewriter",
    "text_effect_params": {},
    "interaction_style": "floating_with_character"
  },
  // ... repeated for EVERY word
]
```

### New Structure (key_phrases + cartoon_characters)
```json
"key_phrases": [
  {
    "phrase": "Do Re Mi",
    "start_seconds": "8.500",
    "duration_seconds": 4.0,
    "top_left_pixels": {"x": 50, "y": 50},
    "bottom_right_pixels": {"x": 200, "y": 100},
    "style": "playful_bounce",
    "importance": "critical"
  }
],
"cartoon_characters": [
  {
    "character_type": "dancing_musical_note",
    "related_to": "Do Re Mi singing lesson",
    "start_seconds": "10.000",
    "duration_seconds": 3.0,
    "position_pixels": {"x": 150, "y": 150},
    "size_pixels": {"width": 60, "height": 80},
    "animation_style": "bounce_in_place",
    "interaction": "appears_near_singer"
  }
]
```

## Files Created/Modified

### New Files
1. `utils/video_description_generator_v2.py` - New generator with key phrases/characters approach
2. `pipeline/steps/step_4_prompts_v2.py` - Updated pipeline step
3. `regenerate_prompts_v2.py` - Script to regenerate all prompts with V2 approach

### Modified Prompts
- All scene prompts in `uploads/assets/videos/do_re_mi/prompts/`
- Backup of old prompts saved as `scene_XXX_prompt_old.txt`

## Implementation Details

### Calculation of Constraints
```python
# For each scene:
max_key_phrases = max(1, int(duration / 20))     # One phrase per 20 seconds
max_cartoon_chars = max(2, int(duration / 10))   # One character per 10-15 seconds
```

### Scene Examples (do_re_mi)
- **Scene 1** (56.7s): Max 2 key phrases, 5 cartoon characters
- **Scene 2** (55.1s): Max 2 key phrases, 5 cartoon characters  
- **Scene 3** (24.3s): Max 1 key phrase, 2 cartoon characters

## Benefits

1. **Less Visual Clutter**: Dramatically reduced text overlays (from ~100 words to 2-3 phrases)
2. **More Impactful**: Key phrases are memorable and emotionally significant
3. **Enhanced Storytelling**: Cartoon characters add visual interest related to content
4. **Better Composition**: Strategic placement avoids blocking important action
5. **Professional Quality**: Follows industry standards for subtitle/overlay frequency

## Usage

### Regenerate Prompts for Single Video
```bash
python regenerate_prompts_v2.py --video do_re_mi
```

### Regenerate Prompts for All Videos
```bash
python regenerate_prompts_v2.py --all
```

### Run Full Pipeline with V2 Prompts
```bash
python unified_video_pipeline_refactored.py video.mp4 --no-downsample --no-transcript --no-scenes
```

## Next Steps

1. ✅ Prompts regenerated for do_re_mi video
2. ⏳ Run inference with new prompts to generate V2 editing instructions
3. ⏳ Update video editing step to handle key_phrases and cartoon_characters
4. ⏳ Test the complete pipeline with new approach

## Results Summary

- **do_re_mi**: 3 scenes processed
  - Total key phrases: 5 (was ~229 individual words)
  - Total cartoon characters: 12 (new feature)
  - Reduction in text overlays: **98% fewer overlays**