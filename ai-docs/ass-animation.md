# Advanced ASS (Advanced SubStation Alpha) Subtitle Format: A Comprehensive Markdown Guide

**ASS (Advanced SubStation Alpha)** is a feature-rich subtitle format supporting intricate styling, animation, effects, and precise positioning. This guide provides a deep-dive into the format structure, style system, override tags, and all supported visual and animation effects with accurate syntax and examples.

***

## Table of Contents
1. [ASS File Structure](#ass-file-structure)
2. [Style System and Fields](#style-system-and-fields)
3. [Events and Dialogue Line Syntax](#events-and-dialogue-line-syntax)
4. [Override Tags (Styling/Effects)](#override-tags-stylingeffects)
   - [Text Styling](#text-styling)
   - [Color and Opacity (Alpha) Controls](#color-and-opacity-alpha-controls)
   - [Text and Layer Geometry](#text-and-layer-geometry)
   - [Line Placement & Animation](#line-placement--animation)
   - [Effects: Fade/Karaoke/Clip](#effects-fadekaraokeclip)
   - [Drawing and Vector Graphics](#drawing-and-vector-graphics)
   - [Special Characters](#special-characters)
5. [Karaoke and Animation Examples](#karaoke-and-animation-examples)
6. [Advanced Features](#advanced-features)
7. [References](#references)

***

## ASS File Structure

ASS files contain three core sections:

```plaintext
[Script Info]
; Metadata and config for the script: video resolution, comments, title, etc.

[V4+ Styles]
; Style definitions (font, color, shadow, outline, alignment, etc.)

[Events]
; Subtitle lines — their timecodes, assigned style, text, effects, etc.
```

### Example Skeleton

```plaintext
[Script Info]
Title: My Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,2,2,30,30,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0000,0000,0000,,{\\b1\\c&H00FF00&}Hello, world!
```


***

## Style System and Fields

**Styles** define the visual look for lines. Key fields:

| Field           | Description                                                     |
|-----------------|-----------------------------------------------------------------|
| Name            | Reference name for the style                                    |
| Fontname        | Typeface                                                        |
| Fontsize        | Size in points, relative to PlayResY                            |
| PrimaryColour   | Main text color (format: `&HBBGGRR&`)                           |
| SecondaryColour | Used in karaoke highlighting                                    |
| OutlineColour   | Border color                                                    |
| BackColour      | Shadow color                                                    |
| Bold            | -1 (true) or 0 (false)                                          |
| Italic          | -1 (true) or 0 (false)                                          |
| Underline       | -1 (true) or 0 (false)                                          |
| StrikeOut       | -1 (true) or 0 (false)                                          |
| ScaleX/Y        | % scaling, typically 100                                        |
| Spacing         | Extra space between letters                                     |
| Angle           | Text rotation                                                   |
| BorderStyle     | 1: Outline+Drop shadow, 3: Opaque box background                |
| Outline         | Outline thickness                                               |
| Shadow          | Shadow depth                                                    |
| Alignment       | Numeric keypad style (1-9, see below)                           |
| MarginL/R/V     | Margins (pixels), L=left, R=right, V=vertical                   |
| Encoding        | Charset encoding                                                |

***

## Events and Dialogue Line Syntax

**Dialogue lines** in `[Events]` block:

```plaintext
Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
```

* Layer: rendering layer (higher renders atop lower)
* Start/End: Timecodes (`H:MM:SS.DD`)
* Style: Style name as defined above
* Name: Speaker (optional)
* MarginL/R/V: Left/Right/Vertical margin overrides (4 digits, `0000` for default)
* Effect: Special effect
* Text: Subtitle with override tags (see below)

***

## Override Tags (Styling/Effects)

**Override tags** appear inside curly braces `{}` within the `Text` field. They allow per-line/per-syllable effects, overriding Style settings.

### Text Styling

| Tag        | Syntax                  | Example                        | Description                             |
|------------|-------------------------|--------------------------------|-----------------------------------------|
| Bold       | `\b1`/`\b0`/`\b<weight>`| `{\\b1}Bold{\\b0}Normal`       | On, off, or set weight (100-900)        |
| Italic     | `\i1`/`\i0`             | `{\\i1}Italic{\\i0}`           | Italic on/off                           |
| Underline  | `\u1`/`\u0`             | `{\\u1}Underlined{\\u0}`       | Underline on/off                        |
| Strikeout  | `\s1`/`\s0`             | `{\\s1}Strikeout{\\s0}`        | Strikeout on/off                        |
| Font Name  | `\fn<name>`             | `{\\fnArial}`                  | Change font                             |
| Font Size  | `\fs<size>`             | `{\\fs24}`                     | Size in points                          |
| Spacing    | `\fsp<space>`           | `{\\fsp10}`                    | Extra pixels between letters            |

### Color and Opacity (Alpha) Controls

| Tag         | Syntax                                     | Example                             | Description                  |
|-------------|--------------------------------------------|-------------------------------------|------------------------------|
| Set Color   | `\c&HBBGGRR&`/`\1c&HBBGGRR&` (primary)     | `{\\c&H00FF00&}`                    | Text color                   |
|             | `\2c&HBBGGRR&` (secondary), `\3c`, `\4c`   | `{\\3c&HFF0000&}`                   | Border, shadow colors        |
| Set Alpha   | `\alpha&HAA&` (all)                        | `{\\alpha&H80&}`                    | Alpha (00=opaque, FF=transparent) |
|             | `\1a`, `\2a`, `\3a`, `\4a`                 | `{\\1a&HFF&}`                       | Per-component alpha          |

### Text and Layer Geometry

| Tag             | Syntax                                       | Example                           | Description                                   |
|-----------------|----------------------------------------------|-----------------------------------|-----------------------------------------------|
| Border Size     | `\bord<size>`                                | `{\\bord3}`                       | Outline thickness                            |
| X/Y Border      | `\xbord<size>` / `\ybord<size>`              | `{\\xbord3}{\\ybord2}`            | Horizontal/vertical outline sizes             |
| Shadow          | `\shad<size>`                                | `{\\shad2}`                       | Drop shadow offset                            |
| X/Y Shadow      | `\xshad<size>` / `\yshad<size>`              | `{\\xshad2}{\\yshad1}`            | Shadow X/Y offset                             |
| Blur Edge       | `\be<strength>`, `\blur<strength>`           | `{\\blur2.5}`                     | Gaussian blur                                 |
| Scale           | `\fscx<%>` / `\fscy<%>`                      | `{\\fscx150}{\\fscy50}`           | Scale width/height (%)                        |
| Shear           | `\fax<factor>` / `\fay<factor>`              | `{\\fax0.2}{\\fay-0.3}`           | Horizontal/vertical shear (perspective)        |
| Rotation        | `\frz<deg>` (z), `\frx<deg>`, `\fry<deg>`    | `{\\frz45}`                       | Rotation around axes                          |
| Letter Spacing  | `\fsp<spacing>`                              | `{\\fsp4}`                        | Change letter spacing                         |

### Line Placement & Animation

| Tag        | Syntax                                             | Example                             | Description                                                    |
|------------|----------------------------------------------------|-------------------------------------|----------------------------------------------------------------|
| Alignment  | `\an<NUM>`                                         | `{\\an7}`                           | 1-9 numpad style (3=bottom-right, 5=center, etc.)              |
| Position   | `\pos(X,Y)`                                        | `{\\pos(320,200)}`                  | Place line at (X,Y) in script coordinates                      |
| Move       | `\move(x1,y1,x2,y2[,t1,t2])`                       | `{\\move(320,240,480,360,0,1000)}`  | Move line from x1,y1 to x2,y2, optionally from t1 to t2 ms      |
| Origin     | `\org(X,Y)`                                        | `{\\org(0,0)}`                      | Sets origin for rotation/transformation                        |
| Wrap Style | `\q<num>`                                          | `{\\q2}`                            | 0,1,2,3: how to wrap lines                                     |
| Reset      | `\r` or `\r<stylename>`                            | `{\\r}` or `{\\rAltStyle}`          | Restore default or named style                                 |

### Effects: Fade/Karaoke/Clip

| Tag      | Syntax                                         | Example                                  | Description                                 |
|----------|------------------------------------------------|------------------------------------------|---------------------------------------------|
| Fade     | `\fad(fadein,fadeout)`                         | `{\\fad(500,300)}`                       | Fade in/out (ms)                            |
| Complex  | `\fade(a1,a2,a3,t1,t2,t3,t4)`                  | `{\\fade(255,32,224,0,500,2000,2200)}`   | 5-phase fade: alpha and time segments        |
| Karaoke  | `\k<dur>`/`\K<dur>`/`\kf<dur>`/`\ko<dur>`      | `{\\k50}`                                | Karaoke timing (centiseconds per syllable)   |
| Animation| `\t([t1,t2[,accel]],MODS)`                     | `{\\t(0,1000,\\bord4\\shad0)}`           | Animate properties over a time range         |
| Clip     | `\clip(x1,y1,x2,y2)`/vector clip               | `{\\clip(0,0,1920,400)}`                 | Rectangular or vector-based mask             |
| Inverse  | `\iclip(...)`                                  | `{\\iclip(0,0,200,100)}`                 | Inverse clipping mask                        |

### Drawing and Vector Graphics

| Tag      | Syntax                               | Example                                                | Description                                  |
|----------|--------------------------------------|--------------------------------------------------------|----------------------------------------------|
| Drawing  | `\p<n>` (n > 0: enable), `\p0` off   | `{\\p1}m 0 0 l 100 0 100 100 0 100{\\p0}`              | Switches to drawing mode, commands follow    |
| Baseline Offset | `\pbo<offset>`               | `{\\pbo-10}`                                           | Vertical offset of the baseline              |
| Drawing Commands | `m,x,y`/`l,x,y`/`b,x1,y1...`| See vector examples below                              | Create lines, beziers, curves, shapes        |

**Vector shapes follow `m`ove, `l`ine, `b`ézier, `s`pline, etc. See the [Aegisub tags ref] for a complete command set.**[1]

### Special Characters

| Character | Syntax | Description               |
|-----------|--------|--------------------------|
| Hard Break| `\N`   | Forced new line          |
| Soft Break| `\n`   | Break if wrapping mode=2 |
| Hard Space| `\h`   | Non-breaking space       |

***

## Karaoke and Animation Examples

### 1. Karaoke Effects

```plaintext
Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0000,0000,0000,,{\\k50}He{\\k30}llo{\\k50} {\\k80}world!
```
Each `{\\kXX}` before a syllable marks the duration (centiseconds) for highlighting during playback.

**Advanced Animation with Karaoke Templater**:
```plaintext
{\\r\\t($start,$mid,\\fscy120)\\t($mid,$end,\\fscy100)}Some lyrics
```
This template grows each syllable vertically at first, then shrinks it back.[2][3]

### 2. Per-Word Animation

```plaintext
{\\fscx20\\fscy20\\t(0,100,\\fscx120\\fscy120)\\t(100,200,\\fscx100\\fscy100)}Bounce!
```
Starts much smaller, grows to 120%, and returns to normal size—bounce effect for attention.[4]

### 3. Moving and Transforming Text

```plaintext
{\\move(320,240,600,800,0,2000)\\frz360\\t(0,2000,\\c&H00FF00&)}Fly By!
```
Moves from (320,240) to (600,800) over 2 seconds, rotating and fading to green along the way.

***

## Advanced Features

- **Layering**: Overlap multiple lines for stacking colored or visual elements
- **Clipping & Masking**: Use drawn or rectangle `\clip` for masking text creatively
- **Drawing Mode**: Create shapes, underlines, logos with vector graphics syntax
- **Transform Chaining**: Multiple `\t` tags can be chained for complex choreography
- **Per-component Coloring**: Independently style text color, border, shadow

***

## References

- [Aegisub Official ASS Override Tags Guide](https://aegisub.org/docs/latest/ass_tags/) — Full grammar, every tag, complex examples[1]
- [ASS Format Full Specification](http://www.tcax.org/docs/ass-specs.htm)[5]
- [In-depth Subtitle Formatting Guide at Matesub](https://matesub.com/resources/subtitle-file-formats)[6]
- [Karaoke Animation Templates](https://aegisub.org/docs/latest/automation/karaoke_templater/tutorial_1/)[2]
- [Shotstack/Community on Animation Features](https://community.shotstack.io/t/feature-request-support-for-ass-subtitle-format-with-advanced-styling-and-animation/757)[7]

***

**This cheat sheet provides the correct syntax and best practices for every ASS subtitle feature, making it suitable for both basic typesetting and advanced TikTok/YouTube kinetic text. For more complex animation, consult the official Aegisub tag documentation.**[3][5][6][7][2][1]

[1](https://aegisub.org/docs/latest/ass_tags/)
[2](https://aegisub.org/docs/latest/automation/karaoke_templater/tutorial_1/)
[3](https://yukisubs.wordpress.com/wp-content/uploads/2017/05/aegisub-3-2-manual.pdf)
[4](https://www.reddit.com/r/ffmpeg/comments/17k0ni7/trying_to_add_animation_to_my_subtitles/)
[5](http://www.tcax.org/docs/ass-specs.htm)
[6](https://matesub.com/resources/subtitle-file-formats)
[7](https://community.shotstack.io/t/feature-request-support-for-ass-subtitle-format-with-advanced-styling-and-animation/757)
[8](https://www.nikse.dk/subtitleedit/formats/assa-override-tags)
[9](https://aegisub.org/docs/latest/styles/)
[10](https://stackoverflow.com/questions/78546868/coordinates-with-file-format-ass)
[11](https://stackoverflow.com/questions/76848089/in-advanced-substation-alpha-ass-file-how-can-i-animate-each-word-as-it-is-spo)
[12](https://pythonhosted.org/pysubs2/api-reference.html)
[13](https://wiki.multimedia.cx/index.php/SubStation_Alpha)
[14](https://www.videomenthe.com/guide-subtitles-formats)
[15](https://github.com/libass/libass)
[16](https://subzap.ai/wiki/subtitles/advancedsubstation)
[17](https://news.ycombinator.com/item?id=25907225)


Here is a Markdown table showing popular **disappearance effects** for subtitles using the ASS (Advanced SubStation Alpha) format, with example tags and explanations:[1][2][3]

## Subtitle Disappearance Effects in ASS

| Effect        | ASS Tag Example              | Explanation                                           |
|---------------|-----------------------------|-------------------------------------------------------|
| Fade out      | `{\fad(0,700)}`             | Fades subtitle out over 700 ms [1][3]        |
| Custom fade   | `{\fade(255,0,0,255,0,700,0,700)}` | Precise fade out control with opacity [1]    |
| Blur dissolve | `{\blur3\fad(0,700)}`       | Fades out while blurring text [2][1]         |
| Color fade    | `{\t(0,700,\c&H808080&)}`   | Changes color as it disappears [1]                |
| Glow dissolve | `{\be2\fad(0,700)}`         | Adds glow effect while fading out [2]            |
| Slide out     | `{\move(640,400,800,400,0,700)}` | Slides subtitle out horizontally over 700 ms [1] |

Each approach combines effects for smoother disappearance, but graphics-based effects like fog or smudge aren't supported in native ASS tags.[2][3]

[1](https://aegisub.org/docs/latest/ass_tags/)
[2](https://www.reddit.com/r/Filmmakers/comments/sobx9p/tutorial_adding_glow_to_ass_subtitles_files/)
[3](https://www.abyssale.com/blog/how-to-change-the-appearances-of-subtitles-with-ffmpeg)