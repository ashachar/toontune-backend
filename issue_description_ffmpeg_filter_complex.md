# FFmpeg Filter Complex Issue - Multiple Text Overlays and Effects

## Problem Description
I'm trying to apply multiple text overlays and visual effects to a video using FFmpeg's filter_complex, but I'm getting syntax errors when trying to chain multiple drawtext filters together with other effects.

## Current Setup
- **FFmpeg Version**: 7.1.1
- **Platform**: macOS (Apple Silicon)
- **Video Input**: MP4 file, H.264, 256x144 resolution, 54.8 seconds duration

## What I'm Trying to Achieve
Apply the following effects to a video in a single FFmpeg command:
1. Multiple text overlays at different timestamps (lyrics/subtitles)
2. Visual effects (bloom, zoom, light sweep) at specific timestamps
3. Debug text overlays showing when effects are active (for test mode)

## The Command That's Failing

```python
# Building filter complex string
filters = []

# Example text overlay filter
filter_str = (
    f"drawtext=text='Let\\'s':"
    f"x=10:y=90:"
    f"fontsize=24:fontcolor=white:"
    f"box=1:boxcolor=black@0.5:boxborderw=5:"
    f"enable='between(t,2.779,3.579)'"
)
filters.append(filter_str)

# Multiple similar filters added...
# Then joined with commas
filter_complex = ",".join(filters)

# FFmpeg command
cmd = [
    "ffmpeg", "-y",
    "-i", video_path,
    "-filter_complex", filter_complex,
    "-c:v", "libx264",
    "-preset", "medium",
    "-crf", "23",
    "-c:a", "copy",
    "-pix_fmt", "yuv420p",
    output_path
]
```

## The Error Message

```
[AVFilterGraph @ 0x600003ae8000] No such filter: '2.779'
Failed to set value 'drawtext=text='Let\'s':x=10:y=90:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:enable='between(t,2.779,3.579)',drawtext=text='start':x=65:y=90:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:enable='between(t,3.579,4.079)',...' for option 'filter_complex': Filter not found
Error parsing global options: Filter not found
```

## Example of the Full Filter String Being Generated

```
drawtext=text='Let\'s':x=10:y=90:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:enable='between(t,2.779,3.579)',drawtext=text='start':x=65:y=90:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:enable='between(t,3.579,4.079)',drawtext=text='beginning':x=10:y=90:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5:enable='between(t,5.28,6.42)',curves=all='0/0 0.5/0.6 1/1':enable='gte(t,2.0)*lte(t,4.0)'
```

## What I've Tried

1. **Using -vf instead of -filter_complex**: This works for simple cases but fails when combining many filters
2. **Escaping colons and commas**: Tried `\\:` and `\\,` but this creates other issues
3. **Using backslashes for escaping**: `drawtext=text='Let\\'s'\\:x=10\\:y=90...` but still fails

## Specific Questions

1. **What's the correct way to escape special characters** (apostrophes, colons, commas) in drawtext filters when using filter_complex?

2. **How should multiple drawtext filters be chained together?** Should they be:
   - Separated by commas: `drawtext=...,drawtext=...,drawtext=...`
   - Chained with semicolons: `drawtext=...;drawtext=...;drawtext=...`
   - Using filter graph notation: `[0:v]drawtext=...[v1];[v1]drawtext=...[v2]`

3. **Is there a better approach for applying many text overlays?** Should I:
   - Use multiple passes with intermediate files?
   - Use subtitles/ASS format instead of drawtext?
   - Split into multiple FFmpeg commands?

4. **How to properly mix drawtext filters with other filters** like curves, zoompan, etc.?

## Working Example Needed

I need a working example that shows how to:
- Apply 10+ text overlays at different timestamps
- Include visual effects (brightness changes, zoom)
- Handle special characters in text (apostrophes, commas, colons)
- All in a single FFmpeg command

## Additional Context

The goal is to create a video with:
- Synchronized lyrics appearing at specific timestamps
- Sound effects (already working separately)
- Visual effects at key moments
- Debug overlays in test mode showing what effects are active

The Python script generates these filters dynamically based on JSON metadata, so the solution needs to work programmatically rather than with hardcoded shell escaping.