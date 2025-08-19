# Unified Video Pipeline Structure

## ✅ Consolidation Complete

There is now **ONE** main entry point for the unified video pipeline:
- `unified_video_pipeline.py` - The single entry point for all video processing

The redundant `unified_video_pipeline_refactored.py` has been archived to `archive/`.

## File Organization

### Main Entry Point
```
unified_video_pipeline.py          # Main script - use this!
```

### Pipeline Components (Modular)
```
pipeline/
├── __init__.py                    # Package exports
├── core/
│   ├── config.py                  # Configuration dataclass
│   └── pipeline.py                # Main orchestrator
└── steps/
    ├── step_1_downsample.py       # Video downsampling
    ├── step_2_transcripts.py      # Transcript generation
    ├── step_3_scenes.py           # Scene splitting
    ├── step_4_prompts.py          # Original prompts (word-by-word)
    ├── step_4_prompts_v2.py       # V2 prompts (key phrases & cartoons)
    ├── step_5_inference.py        # LLM inference
    ├── step_6_edit_videos.py      # Video editing
    └── step_7_karaoke.py          # Karaoke generation
```

## Directory Structure Created

The pipeline creates this structure for each video:

```
uploads/assets/videos/{video_name}/
├── video.mp4                      # Original video
├── video_downsampled.mp4          # Downsampled full video
├── transcripts/
│   ├── transcript_sentences.json
│   └── transcript_words.json
├── scenes/
│   ├── original/                  # Full resolution scenes
│   │   └── scene_*.mp4
│   ├── downsampled/              # Downsampled scenes (PARALLEL to original/)
│   │   └── scene_*.mp4
│   ├── edited/                   # Edited with effects
│   │   └── scene_*.mp4
│   └── karaoke/                  # Karaoke versions
│       └── scene_*.mp4
├── prompts/
│   └── scene_*_prompt.txt        # V2: key phrases & cartoon characters
├── inferences/
│   └── scene_*_inference.json
└── metadata/
    └── pipeline_state.json
```

## Usage Examples

### Basic Usage
```bash
# Process a new video completely
python unified_video_pipeline.py my_video.mp4

# Dry run (skip LLM inference)
python unified_video_pipeline.py my_video.mp4 --dry-run
```

### Skip Specific Steps
```bash
# Skip downsample and transcript (use existing)
python unified_video_pipeline.py my_video.mp4 --no-downsample --no-transcript

# Only run inference and editing (prompts already exist)
python unified_video_pipeline.py my_video.mp4 \
  --no-downsample --no-transcript --no-scenes --no-prompts
```

### Feature Flags
```bash
# Generate karaoke version
python unified_video_pipeline.py my_video.mp4 --karaoke

# Test mode with effect labels
python unified_video_pipeline.py my_video.mp4 --test-mode
```

## V2 Prompts (Key Phrases & Cartoons)

The V2 prompt system replaces word-by-word overlays with:
- **Key phrases**: Max 4 words, once per 20 seconds
- **Cartoon characters**: Related to content, once per 20 seconds
- **Non-overlapping**: Key phrases and cartoons never appear simultaneously

### To Use V2 Prompts
```bash
# Regenerate prompts with V2 approach
python regenerate_prompts_v2.py --video do_re_mi

# Then run the pipeline
python unified_video_pipeline.py uploads/assets/videos/do_re_mi/video.mp4 \
  --no-downsample --no-transcript --no-scenes --no-prompts
```

## Key Points

1. **Single Entry Point**: Always use `unified_video_pipeline.py`
2. **Correct Directory Structure**: `downsampled/` is parallel to `original/` and `edited/`
3. **V2 Prompts**: Dramatically reduce text overlays (98% fewer)
4. **Modular Design**: Each step can be skipped independently

## Common Workflows

### New Video
```bash
python unified_video_pipeline.py new_video.mp4
```

### Re-run with Changes
```bash
# After updating prompts
python unified_video_pipeline.py video.mp4 \
  --no-downsample --no-transcript --no-scenes --no-prompts

# After updating inference results
python unified_video_pipeline.py video.mp4 \
  --no-downsample --no-transcript --no-scenes --no-prompts --no-inference
```

### Test Changes
```bash
# Test mode to see effect labels
python unified_video_pipeline.py video.mp4 --test-mode --no-downsample --no-transcript
```