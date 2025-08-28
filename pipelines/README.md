# Golden Standard Pipelines

This folder contains fully working, production-ready pipeline examples that demonstrate best practices for various video processing and animation tasks.

## Current Pipelines

### word_level_pipeline.py
**Word-by-Word Text Animation with Fog Dissolve**

A complete implementation showing:
- Individual word object management throughout all animation phases
- Fixed position calculation (positions never change after initial calculation)
- Sentence-level direction consistency (all words from same direction)
- Smooth fog dissolve transitions without position shifts
- Audio preservation during video processing
- Proper easing functions and timing

Key features:
- Uses `WordObject` dataclass for persistent word state
- Sine-based easing for smooth animations
- Per-word fog parameters (randomized once)
- Dissolved state tracking to prevent reappearance
- FFmpeg audio merging for final output

Usage:
```python
python pipelines/word_level_pipeline.py
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