# Golden Standard Pipelines

This folder contains fully working, production-ready pipeline examples that demonstrate best practices for various video processing and animation tasks.

## Current Pipelines

### word_level_pipeline.py (Multi-line Version)
**Complete Word-by-Word Text Animation with Multi-line Support**

A production-ready implementation with automatic line wrapping for long sentences:
- **Maximum 6 words per row** for optimal readability
- **Automatic line wrapping** for sentences exceeding 6 words
- **Proper vertical spacing** between rows
- **Row-based animation stagger** for visual appeal
- Parses full transcript JSON into sentence segments
- Individual word object management throughout all animation phases  
- Fixed position calculation (positions never change after initial calculation)
- Sentence-level direction consistency (all words from same direction)
- Alternating direction between sentences for visual variety
- Smooth fog dissolve transitions without position shifts
- Audio preservation during video processing

Key features:
- `MultiLineWordPipeline` class with configurable max words per row
- Row tracking for each word object
- Automatic sentence splitting into multiple rows
- Centered multi-line text layout
- Row-based stagger in rise animations
- All rows of a sentence dissolve together
- Works from any directory location

Usage:
```python
python pipelines/word_level_pipeline.py
```

Output: `outputs/ai_math1_multiline_30s_h264.mp4` (30-second test)

### word_level_pipeline_single_line.py
**Single-line Word Animation (Full Video)**

The previous version that keeps all words on a single line.
Processes entire transcript but may have readability issues with long sentences.

Usage:
```python
python pipelines/word_level_pipeline_single_line.py
```

Output: `outputs/ai_math1_full_word_animation_h264.mp4`

### word_level_pipeline_simple.py (Original 6-second Demo)
**Simple Word-by-Word Animation Demo**

The original 6-second demonstration with two hardcoded sentences.
Useful for quick testing and understanding the core concepts.

Usage:
```python
python pipelines/word_level_pipeline_simple.py
```

Output: `outputs/word_level_pipeline_h264.mp4`

## Pipeline Development Guidelines

When creating new pipelines for this folder:

1. **Self-Contained**: Pipeline should run without external dependencies beyond standard imports
2. **Well-Documented**: Include clear docstrings and comments explaining the approach
3. **Best Practices**: Follow all rules from CLAUDE.md and post_production_effects_guide.md
4. **Audio Preservation**: Always maintain audio tracks when processing video
5. **H.264 Output**: Ensure final video is properly encoded for compatibility
6. **Error Handling**: Include proper error handling and user feedback
7. **Modular Design**: Structure code for reusability and clarity

## Reference Documentation

- See `docs/post_production_effects_guide.md` for detailed animation rules and guidelines
- See `CLAUDE.md` for project-wide coding standards and requirements
- Check `utils/animations/` for reusable animation components