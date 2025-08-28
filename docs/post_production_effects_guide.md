# Post-Production Effects Guide

## Background Replacement with Stock Videos

### Overview
The system can automatically replace video backgrounds with stock footage from Coverr.co or other sources. It uses AI-powered segmentation to extract foreground subjects and composite them over new backgrounds.

### Key Components

#### 1. **CoverrManager** (`utils/video/background/coverr_manager.py`)
- Manages stock video downloads from Coverr API
- Intelligently caches videos to avoid re-downloading
- Uses AI to select appropriate backgrounds based on transcript keywords
- Falls back to demo gradients if API unavailable

**Key Features:**
- Keyword extraction from transcripts
- Smart video selection using OpenAI
- Local caching in `assets/videos/coverr/`
- Project-specific caching in `uploads/assets/videos/{project_name}/`

#### 2. **BackgroundReplacer** (`utils/video/background/replace_background.py`)
- Handles the actual background replacement process
- Uses SAM2 for foreground mask extraction
- Composites using FFmpeg maskedmerge filter
- Supports both mask-based and chromakey methods

**Processing Pipeline:**
1. Extract foreground masks from original video
2. Download/retrieve background stock video
3. Composite using FFmpeg with mask video
4. Output H.264 encoded result

### Usage Examples

#### Using Claude Subagent for Coverr Integration

To fetch real stock videos from Coverr API, use the Claude subagent:

```bash
# Use Claude's Task tool to fetch stock videos
claude> Use the Task tool with subagent_type="general-purpose" to:
1. Check COVERR_KEY in .env
2. Search Coverr for videos matching transcript keywords
3. Download appropriate stock videos to project folder
4. Return list of downloaded videos with timestamps
```

The subagent will:
- Authenticate with Coverr API using COVERR_KEY from .env
- Search for relevant videos based on transcript content
- Use AI to select the most appropriate videos
- Download and cache videos with proper naming convention
- Handle fallback searches if primary keywords don't yield results

#### Basic Background Replacement (5-second test)
```python
from utils.video.background.replace_background import BackgroundReplacer

replacer = BackgroundReplacer(demo_mode=False)
output = replacer.process_video(
    video_path="uploads/assets/videos/ai_math1_segment_5sec.mp4",
    project_name="ai_math1",
    start_time=0,
    end_time=5,
    use_mask=True  # Required for non-green screen videos
)
```

#### Multiple Timestamps with Different Backgrounds
```python
import json
from pathlib import Path
from utils.video.background.replace_background import BackgroundReplacer
from utils.video.background.coverr_manager import CoverrManager

def add_stock_backgrounds_at_timestamps(video_path, project_name, timestamps):
    """
    Add different stock video backgrounds at specific timestamps.
    
    Args:
        video_path: Path to input video
        project_name: Project name (e.g., "ai_math1")
        timestamps: List of dicts with 'start', 'end', 'keywords'
    """
    replacer = BackgroundReplacer(demo_mode=False)
    manager = CoverrManager()
    
    segments = []
    for i, segment in enumerate(timestamps):
        # Extract segment from original video
        segment_path = f"temp_segment_{i}.mp4"
        extract_cmd = [
            "ffmpeg", "-y",
            "-ss", str(segment['start']),
            "-i", str(video_path),
            "-t", str(segment['end'] - segment['start']),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            segment_path
        ]
        subprocess.run(extract_cmd, check=True)
        
        # Search for background based on keywords
        videos = manager.search_videos(segment['keywords'])
        if videos:
            bg_video = manager.select_best_video(videos, segment.get('context', ''))
            bg_path = manager.download_video(bg_video, project_name, 
                                            Path(f"uploads/assets/videos/{project_name}"),
                                            segment['start'], segment['end'])
            
            # Replace background
            output = replacer.replace_background_ffmpeg(
                Path(segment_path), bg_path,
                Path(f"output_segment_{i}.mp4"),
                mask_video_path=None  # Use existing mask if available
            )
            segments.append(output)
    
    # Concatenate all segments
    # ... concatenation code ...
```

### Pre-calculated Masks and Green Screen

The system can use pre-calculated RVM (Robust Video Matting) outputs:

- **RVM Green Screen**: `ai_math1_rvm_green_5s_024078685789.mp4` - Foreground with green background (RECOMMENDED)
- **RVM Mask**: `ai_math1_rvm_mask_5s_024078685789.mp4` - Often just grayscale video, NOT a true alpha mask

#### CRITICAL LESSON: Verify Your Mask Type

**IMPORTANT**: The RVM "mask" file may NOT be a true alpha mask! Always verify:
1. Extract a frame and check if it's binary (black/white) or grayscale
2. If it's just a grayscale version of the video, use the green screen version instead
3. True alpha masks have distinct foreground (white/255) and background (black/0) regions

#### Recommended Method: Green Screen with Refined Chromakey

After extensive testing, the most reliable method is using the green screen with carefully tuned chromakey:

```python
# BEST PRACTICE: Use green screen with refined chromakey settings
cmd = [
    "ffmpeg", "-y",
    "-stream_loop", "-1", "-i", stock_video,    # Background (looped)
    "-ss", start_time, "-t", duration,
    "-i", green_screen_video,                    # Green screen foreground
    "-filter_complex",
    "[0:v]scale=1280:720,setpts=PTS-STARTPTS[bg];"
    "[1:v]scale=1280:720,setpts=PTS-STARTPTS[fg];"
    # Refined chromakey with tight thresholds
    "[fg]chromakey=green:0.08:0.04[keyed];"     # Lower values = more selective
    "[keyed]despill=type=green:mix=0.2:expand=0[clean];"  # Light despill
    "[bg][clean]overlay=shortest=1[out]",
    "-map", "[out]",
    "-map", "1:a?",  # Preserve audio
    output_path
]
```

#### Chromakey Tuning Guide

**Critical Parameters:**
- **Similarity** (0.08): How close to pure green - LOWER = more selective
- **Blend** (0.04): Edge softness - LOWER = sharper edges
- **Despill mix** (0.2): Color correction strength - LOWER = preserve original colors

**Common Issues & Solutions:**

1. **Green edges around subject**: Decrease similarity (try 0.05-0.08)
2. **Background bleeding through foreground objects**: 
   - Blue/cyan objects may be removed if similarity too high
   - Solution: Use similarity=0.08, blend=0.04
3. **Harsh edges**: Slightly increase blend (but keep below 0.06)
4. **Color shift in foreground**: Reduce despill mix (0.1-0.2)

#### Testing Chromakey Thresholds

Always test different thresholds before final render:

```python
test_configs = [
    {"similarity": 0.05, "blend": 0.02, "name": "very_tight"},
    {"similarity": 0.08, "blend": 0.04, "name": "tight"},  # Usually best
    {"similarity": 0.10, "blend": 0.05, "name": "moderate"},
    {"similarity": 0.12, "blend": 0.06, "name": "loose"},
]
```

#### Complete Pipeline Reference

See `pipelines/apply_stock_backgrounds_refined_chromakey.py` for the production-ready implementation.

### Stock Video Selection Criteria

The CoverrManager uses these strategies to select appropriate backgrounds:

1. **Keyword Extraction**: Analyzes transcript for visual themes
   - AI/tech content → futuristic, digital, circuit backgrounds
   - Math/science → abstract, geometric, particle effects
   - Data/analytics → visualization, network, graph animations

2. **AI Selection**: Uses OpenAI to pick the most contextually appropriate video
3. **Heuristic Fallback**: Scores videos based on keyword matches

### Caching Strategy

Videos are cached at multiple levels:
1. **Global Coverr Cache**: `assets/videos/coverr/`
2. **Project Cache**: `uploads/assets/videos/{project_name}/`
3. **Naming Convention**: `{project}_background_{start}_{end}_{video_id}.mp4`

### Performance Optimization

- **Sample Rate**: Process every Nth frame for mask extraction (default: 5)
- **Batch Processing**: Extract all masks first, then composite
- **Resolution**: Standardize to 1920x1080 for consistency
- **Codec**: Use libx264 with preset=fast, crf=23

### API Configuration

Required in `.env`:
```
COVERR_KEY=your_coverr_api_key
OPENAI_API_KEY=your_openai_key  # Optional, for AI selection
```

### Demo Mode

When `demo_mode=True` or API unavailable:
- Creates animated gradient backgrounds
- Uses FFmpeg's lavfi filters for procedural backgrounds
- No external API calls required

### Common Issues & Solutions

1. **API Authentication Failed**: Check COVERR_KEY in .env
2. **No Videos Found**: Falls back to broader search terms or demo mode
3. **Mask Extraction Slow**: Reduce sample_rate or use pre-calculated masks
4. **Quality Loss**: Adjust CRF value (lower = better quality, larger file)

## Transcript-Synchronized Word Animations (CRITICAL SECTION)

### Core Principles & Rules of Thumb

#### 1. **Word-Level Object Management**
**CRITICAL**: Always maintain individual word objects throughout ALL animation phases
- Create a `WordObject` class with fixed position (x, y) that NEVER changes
- Store per-word timing, effects parameters, and state
- Apply effects to individual words, not entire sentences
- This prevents position shifts and ensures smooth transitions

#### 2. **Direction Consistency Per Sentence**
**RULE**: All words in a sentence MUST enter from the SAME direction
- Decide direction (from below/above) per sentence, not per word
- Alternating can happen between sentences, never within
- Example: Sentence 1 all from below, Sentence 2 all from above

#### 3. **Fixed Position Calculation**
**CRITICAL**: Calculate word positions ONCE and never recalculate
```python
# Calculate all positions upfront
start_x = center[0] - total_width // 2
for word in words:
    word.x = current_x  # Fixed forever
    word.y = center[1]  # Fixed forever
    current_x += word_width + space_width
```

#### 4. **Fog Dissolve Rules**
**IMPORTANT**: Fog dissolve implementation requirements:
- Letters must stay at EXACT positions - only fog effect varies
- Each letter gets random fog parameters ONCE (blur_x, blur_y, density)
- Progressive phases: blur → fog texture → final fade
- After dissolve completes, words must NOT reappear (is_dissolved flag)

#### 5. **Audio Preservation**
**MANDATORY**: Always preserve audio when processing videos
```bash
# Extract with audio
ffmpeg -i input.mp4 -t 6 -c:v libx264 -c:a copy segment.mp4

# Merge audio back after processing
ffmpeg -i processed.mp4 -i original.mp4 -c:v libx264 -c:a copy -map 0:v:0 -map 1:a:0 final.mp4
```

#### 6. **Animation Timing Guidelines**
- **Rise duration**: 700-1200ms for gentle entry
- **Easing function**: Use sine-based `(1 - np.cos(progress * π)) / 2` for smoothness
- **Opacity**: Square the progress for softer fade-in `opacity = eased_progress²`
- **Word spacing**: Based on transcript gaps, not fixed intervals

### Golden Standard Pipeline Example

See `pipelines/word_level_pipeline.py` for the complete implementation following all best practices.

#### Key Implementation Pattern
```python
@dataclass
class WordObject:
    text: str
    x: int  # Fixed X position (never changes)
    y: int  # Fixed Y position (never changes)
    width: int
    height: int
    start_time: float
    end_time: float
    rise_duration: float
    from_below: bool  # Direction for this word's sentence
    # Fog parameters (randomized once, then fixed)
    blur_x: float
    blur_y: float
    fog_density: float
    dissolve_speed: float

# Process each frame
for word_obj in word_objects:
    if is_dissolved:
        continue  # Skip rendering
    if fog_progress > 0:
        apply_fog_to_word()  # Apply fog without position change
    render_word()  # Always at fixed position
```

### Common Pitfalls to Avoid

1. **Never recalculate positions** - This causes words to jump
2. **Don't alternate directions within sentences** - Looks chaotic
3. **Avoid caching frames/masks** - Dynamic content requires fresh calculations
4. **Don't forget audio** - Use `-c:a copy` in FFmpeg commands
5. **Never let dissolved text reappear** - Track dissolved state properly

## Related Effects

### Transcript-Synchronized Word Animations

#### Overview
Titles and text can be synchronized with the video's transcript so each word appears and disappears at its exact spoken timestamp. This creates professional subtitle-like effects where words rise, fade, or transform in perfect sync with speech.

#### Implementation (`utils/animations/3d_animations/word_3d/word_3d.py`)
- **WordRiseSequence3D**: Words rise from below in sequence
- **WordDropIn3D**: Words drop from above with bounce
- **WordWave3D**: Center-outward wave pattern

#### Transcript Timing Integration
```python
# Example: Sync word animation to transcript
from utils.transcript.whisper_transcript import extract_transcript
from utils.animations.3d_animations.word_3d import WordRiseSequence3D

# Extract word-level timestamps
transcript = extract_transcript(video_path, word_timestamps=True)

# Calculate word timing
for word_data in transcript['words']:
    word = word_data['word']
    start_ms = word_data['start'] * 1000  # Convert to milliseconds
    end_ms = word_data['end'] * 1000
    
    # Configure animation to match speech timing
    config = Animation3DConfig(
        text=word,
        duration_ms=end_ms - start_ms,
        position=(640, 360, 0)
    )
    
    # Apply word rise effect at exact timestamp
    animation = WordRiseSequence3D(
        config,
        word_spacing_ms=0,  # No spacing - use transcript timing
        rise_duration_ms=min(500, end_ms - start_ms),  # Adapt to word duration
        fade_in=True
    )
```

#### Key Timing Considerations
1. **Word Entry**: Each word should begin animation at its transcript start time
2. **Word Exit**: Words can either:
   - Stay visible after speaking (accumulation effect)
   - Fade out when the next word starts
   - Disappear after their transcript end time
3. **Animation Duration**: Rise/drop duration should be shorter than word duration
4. **Spacing**: Use transcript gaps, not fixed spacing

#### Example: 5-Second Synchronized Title
```python
def create_transcript_synced_titles(video_path, start_sec, end_sec):
    # Extract transcript for segment
    transcript = extract_transcript(video_path)
    segment_words = [w for w in transcript['words'] 
                     if w['start'] >= start_sec and w['end'] <= end_sec]
    
    # Group into phrases (optional)
    phrases = group_words_into_phrases(segment_words)
    
    # Apply animation with proper timing
    for phrase in phrases:
        text = ' '.join([w['word'] for w in phrase])
        start_ms = phrase[0]['start'] * 1000
        duration_ms = (phrase[-1]['end'] - phrase[0]['start']) * 1000
        
        # Create word-rise animation matching speech rhythm
        animation = WordRiseSequence3D(
            config=Animation3DConfig(text=text, duration_ms=duration_ms),
            word_spacing_ms=calculate_word_gaps(phrase),  # Based on speech gaps
            rise_duration_ms=600,  # Gentle rise
            fade_in=True
        )
```

### Text Behind Object (`utils/animations/text_behind_segment.py`)
- Places text behind moving subjects using dynamic masking
- **CRITICAL**: Never cache masks - always calculate fresh for every frame
- Uses foreground extraction to create occlusion effects

### 3D Text Animation (`utils/animations/apply_3d_text_animation.py`)
- Adds 3D text with optimal positioning
- Auto-finds least occluded location
- Integrates with background masks

### Video Segmentation (`utils/video_segmentation/`)
- Automatic segment detection with SAM2
- AI-powered labeling with Gemini
- Tracks segments throughout video

## File References

Key implementation files:
- Background replacement: `utils/video/background/replace_background.py:192`
- Coverr manager: `utils/video/background/coverr_manager.py:417`
- Segment extraction: `utils/video/segmentation/segment_extractor.py`
- Test script: `test_background_removal.py:57`