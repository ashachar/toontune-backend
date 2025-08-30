# Unified Video Processing Pipeline - Refactored

## Overview
The unified video pipeline has been refactored from a single 1000+ line file into modular components, each under 200 lines, while keeping **100% of the original logic intact**.

## Structure

```
pipeline/
├── __init__.py              # Package initialization
├── core/
│   ├── config.py           # Configuration dataclass (26 lines)
│   └── pipeline.py         # Main orchestrator (148 lines)
└── steps/
    ├── step_1_downsample.py    # Video downsampling (50 lines)
    ├── step_2_transcripts.py   # Transcript generation (171 lines)
    ├── step_3_scenes.py        # Scene splitting (144 lines)
    ├── step_4_prompts.py       # Prompt generation (138 lines)
    ├── step_5_inference.py     # LLM inference (36 lines)
    ├── step_6_edit_videos.py   # Video editing (171 lines)
    └── step_7_karaoke.py       # Karaoke captions (116 lines)
```

## Features Preserved

All original functionality remains intact:

1. **Video Downsampling** - Creates 256x256 versions for processing
2. **Transcript Generation** - Uses OpenAI Whisper API for word-level timing
3. **Scene Splitting** - Intelligent scene boundaries based on transcript
4. **Prompt Generation** - Comprehensive prompts with effects documentation
5. **LLM Inference** - Placeholder for Gemini Pro integration
6. **Video Editing** - Applies effects, sound, and text overlays
7. **Karaoke Generation** - With timestamp interpolation for missing words

## Usage

The pipeline can be used exactly as before:

```bash
# Full pipeline
python unified_video_pipeline.py video.mp4

# With karaoke captions
python unified_video_pipeline.py video.mp4 --karaoke

# Dry run mode
python unified_video_pipeline.py video.mp4 --dry-run

# Skip specific steps
python unified_video_pipeline.py video.mp4 \
  --no-downsample \
  --no-transcript \
  --karaoke
```

## Backward Compatibility

The original `unified_video_pipeline.py` file now redirects to the refactored implementation, ensuring all existing scripts and workflows continue to work without modification.

## Benefits of Refactoring

1. **Maintainability** - Each step is isolated in its own module
2. **Testability** - Individual steps can be tested independently
3. **Readability** - Each file is focused on a single responsibility
4. **Extensibility** - New steps can be added easily
5. **Debugging** - Issues can be traced to specific modules

## Configuration

All pipeline configuration is centralized in `PipelineConfig`:

```python
from pipeline import PipelineConfig, UnifiedVideoPipeline

config = PipelineConfig(
    video_path="video.mp4",
    generate_karaoke=True,
    karaoke_style="continuous",
    test_mode=False
)

pipeline = UnifiedVideoPipeline(config)
pipeline.run()
```

## Directory Structure Created

```
uploads/assets/videos/{video_name}/
├── video.mp4                    # Original video
├── video_downsampled.mp4        # Downsampled version
├── transcripts/
│   ├── transcript_sentences.json
│   └── transcript_words.json
├── scenes/
│   ├── original/               # Original scene files
│   ├── downsampled/           # Downsampled scenes
│   ├── edited/                # Edited with effects
│   └── karaoke/               # With karaoke captions
├── prompts/                    # LLM prompts
├── inferences/                 # LLM responses
└── metadata/
    └── pipeline_state.json     # Pipeline execution state
```