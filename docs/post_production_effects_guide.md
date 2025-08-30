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

### Cached RVM Background Removal System

The system includes intelligent caching for Replicate's Robust Video Matting to avoid unnecessary API calls:

#### Cache Management (`utils/video/background/cached_rvm.py`)

**Key Features:**
- Automatically checks for existing RVM outputs before making API calls
- Stores results in project-specific folders
- Generates consistent hashes for cache identification
- Saves both green screen and mask versions

**Cache Location:**
```
uploads/assets/videos/{video_name}/
├── {video_name}_rvm_green_{duration}s_{hash}.mp4  # Green screen output
├── {video_name}_rvm_mask_{duration}s_{hash}.mp4   # Mask output
└── {video_name}_rvm_meta_{duration}s_{hash}.json  # Metadata
```

**Usage:**
```python
from utils.video.background.cached_rvm import CachedRobustVideoMatting

processor = CachedRobustVideoMatting()

# Will check cache first, only call API if needed
green_screen_path = processor.get_rvm_output(
    video_path="uploads/assets/videos/ai_math1.mp4",
    duration=5  # Process first 5 seconds (None for full video)
)

# Direct compositing with background
final_output = processor.composite_with_background(
    video_path="uploads/assets/videos/ai_math1.mp4",
    background_video="path/to/background.mp4",
    output_path="outputs/composited.mp4",
    duration=5
)
```

**Cache Workflow:**
1. **Check existing cache**: Look for RVM outputs with matching hash
2. **Use cached if found**: No API call needed, instant availability
3. **Process if not cached**: Call Replicate API and save results
4. **Store for future use**: Save green screen, mask, and metadata

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

### Dynamic Background Changes Throughout Video

The system supports changing backgrounds at different timestamps based on content themes:

#### Background Timestamp Allocation

The system uses AI to intelligently allocate backgrounds based on transcript analysis. This is configured in `prompts.yaml` under `transcript_background_allocation`.

**Prompt Features:**
- Analyzes full transcript for theme shifts
- Allocates backgrounds to timestamp ranges
- Ensures minimum 10-second segments (avoids rapid changes)
- Maximum 10 background changes per video
- Returns structured JSON with timestamps, themes, and keywords

**Available Themes:**
- `abstract_tech` - Digital particles, circuits for AI/tech content
- `mathematics` - Equations, geometric shapes for math sections
- `data_visualization` - Charts, networks for data/analytics
- `research` - Laboratory, scientific imagery
- `nature` - Calm scenes for philosophical discussions
- `innovation` - Creative process visuals
- `education` - Learning environment backgrounds
- `cosmic` - Universe imagery for big concepts
- `minimal` - Subtle patterns for focus sections

#### Full Video Pipeline (`pipelines/full_video_stock_backgrounds_pipeline.py`)

**Features:**
- Uses `transcript_background_allocation` prompt to determine segments
- Automatically searches and downloads relevant stock videos
- Changes backgrounds at AI-determined timestamps
- Uses cached RVM to avoid redundant processing

**Example Workflow:**
```python
# Thematic segments automatically identified from transcript
segments = [
    {"start": 0, "end": 15, "theme": "AI Innovation", "keywords": ["AI", "technology"]},
    {"start": 15, "end": 35, "theme": "Mathematics", "keywords": ["calculus", "math"]},
    {"start": 35, "end": 60, "theme": "Data Science", "keywords": ["data", "analytics"]}
]

# Process each segment with appropriate background
for segment in segments:
    # Check cache for existing stock video
    # Download new if needed using Coverr API
    # Apply chromakey with segment timing
    apply_chromakey_segment(green_screen, stock_video, output, 
                           segment['start'], segment['duration'])
```

#### Quick Demo (`apply_dynamic_backgrounds_5sec.py`)

Demonstrates multiple background changes in a 5-second video:
```bash
python apply_dynamic_backgrounds_5sec.py
```

Output shows backgrounds changing every 1-2 seconds based on content.

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

#### 1. **Centralized Background Cache** (`uploads/assets/backgrounds/`)
**PRIMARY CACHE - Check here first before downloading new backgrounds!**

Managed by `utils/video/background/background_cache_manager.py`:
- **Universal cache** for all background videos across projects
- **Searchable by theme and keywords**
- **Metadata tracking** with source, duration, and relevance scoring
- **Import existing backgrounds** from project folders

**Cache Structure:**
```
uploads/assets/backgrounds/
├── cache_metadata.json         # Searchable metadata for all backgrounds
├── mathematics_f55227553cc3.mp4       # Theme-based naming
├── ai_visualization_2f90f56df74c.mp4
├── data_analytics_e4b6ba46605b.mp4
└── [theme]_[hash].mp4
```

**Usage:**
```python
from utils.video.background.background_cache_manager import BackgroundCacheManager

manager = BackgroundCacheManager()

# Search for existing backgrounds BEFORE downloading
existing = manager.get_best_match(
    theme="mathematics",
    keywords=["calculus", "equations", "geometry"]
)

if existing:
    print(f"Found cached background: {existing}")
    # Use this instead of downloading
else:
    # Only download if not in cache
    # Then add to cache after downloading
    manager.add_background(
        downloaded_video,
        theme="mathematics",
        keywords=["calculus", "equations"],
        source="coverr",
        source_id="abc123"
    )
```

**Key Features:**
- **Theme-based search**: Find backgrounds by visual theme
- **Keyword matching**: Score-based relevance ranking
- **Automatic deduplication**: Hash-based to prevent duplicates
- **Import tool**: `import_existing_backgrounds.py` to consolidate existing videos

#### 2. **Project-Specific Cache** (`uploads/assets/videos/{project_name}/`)
- Contains project-specific videos and RVM outputs
- Naming: `{project}_background_{start}_{end}_{video_id}.mp4`

#### 3. **Legacy Coverr Cache** (`assets/videos/coverr/`)
- Original Coverr downloads (being migrated to centralized cache)
- Can be imported using `manager.import_from_coverr_cache()`

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

### Foreground Masking with Cached RVM

The word-level pipeline now integrates with the cached RVM (Robust Video Matting) system for accurate foreground masking when rendering text behind subjects.

#### How It Works

1. **Automatic Cache Detection**: When initializing the pipeline, it checks for existing RVM masks in the video's project folder
2. **Mask Types Supported**:
   - **RVM Mask Videos**: `{video_name}_rvm_mask_*.mp4` - Direct mask usage
   - **RVM Green Screen Videos**: `{video_name}_rvm_green_*.mp4` - Converts to mask on-the-fly
3. **Fallback Method**: If no cached mask exists, falls back to edge detection (not recommended)

#### Usage in Word-Level Pipeline

The masking system is automatically initialized when you run the word-level pipeline:

```python
from pipelines.word_level_pipeline import create_word_level_video

# The pipeline automatically looks for cached masks
create_word_level_video("uploads/assets/videos/ai_math1.mp4", duration_seconds=6.0)
```

#### Pre-generating Masks

To ensure high-quality masking, pre-generate RVM masks before running the pipeline:

```python
from utils.video.background.cached_rvm import CachedRobustVideoMatting

processor = CachedRobustVideoMatting()

# Generate mask for the video (will be cached)
green_screen_path = processor.get_rvm_output(
    "uploads/assets/videos/ai_math1.mp4",
    duration=None  # Process full video
)
```

#### Implementation Details

The masking is handled by `ForegroundMaskExtractor` in `pipelines/word_level_pipeline/masking.py`:

- **Initialization**: Accepts video path to locate cached masks
- **Frame Synchronization**: Uses frame numbers to sync mask video with main video
- **Green Screen Conversion**: Automatically converts green screen to binary mask
- **Quality Assurance**: Clean, smooth masks without the noise of edge detection

#### Benefits Over Edge Detection

- **Accurate Segmentation**: AI-powered person detection vs noisy edge detection
- **Clean Masks**: No random pixels or artifacts
- **Temporal Consistency**: Mask is consistent across frames
- **Better Text Rendering**: Text behind person looks natural without cut-up letters

## Transcript-Synchronized Word Animations (CONTINUED)

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

#### 7. **Multi-line Text Layout**
**IMPORTANT**: Long sentences should wrap to multiple lines for readability
- **Maximum 6 words per row** - prevents overcrowding
- **Automatic line wrapping** - split sentences exceeding max words
- **Centered layout** - each row centered horizontally
- **Vertical spacing** - 1.4x font size between rows
- **Row-based stagger** - slight timing offset for each row during animation
- **Unified dissolve** - all rows of a sentence dissolve together

#### 8. **Enriched Transcripts with AI Analysis**
**CRITICAL**: Use LLM to analyze transcript importance for dynamic emphasis
- **Sub-sentence grouping** - Break transcript into meaningful 1-6 word phrases
- **Importance scoring** - Assign 0.0-1.0 scores based on content significance
- **Emphasis types**:
  - `title` - Main topics, headlines (1.4x size, bold, golden, raised)
  - `critical` - Key insights, numbers (1.25x size, bold, red tint, glow)
  - `important` - Supporting concepts (1.15x size, conditional bold)
  - `normal` - Regular narrative (base size)
- **Visual properties per phrase**:
  ```python
  visual_style = {
      "font_size_multiplier": 1.0-1.4,
      "bold": True/False,
      "position_y_offset": -30 to 0,
      "color_tint": (r, g, b),
      "animation_speed": 0.7-1.0,
      "glow_effect": True/False
  }
  ```
- **LLM prompt structure** - Analyze transcript → identify key concepts → assign scores
- **Mock fallback** - Generate reasonable emphasis without API using keyword heuristics

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