# /summarize

Intelligently collect relevant project files for an issue and create an XML context document.

## Usage
```
/summarize "full issue description including proposed solutions"
```

## Description
**CRITICAL RULES:**
1. **Claude MUST decide which files are relevant, NOT the Python script!**
2. **Pass the ENTIRE user input to the script, not just the problem description!**
3. **Claude MUST decide on an issue prefix** for debug logging (e.g., "ANIM_HANDOFF", "TEXT_POS", "DISSOLVE_FIX")

When this command is invoked, Claude will:
1. **Capture the COMPLETE user input** - problem, proposed solutions, hints, directions, everything
2. **Analyze the issue** to understand what the problem is about
3. **Decide on a short, descriptive issue prefix** for debug logging (5-15 characters, all caps with underscores)
4. **Search for relevant files** using Grep, Glob, and other tools to find ONLY directly relevant code files
5. **Curate a focused list** of the most relevant files (typically 5-20 files max)
6. **Call the XML generation script** with the FULL user input (not summarized), issue prefix, and specific file paths

The Python script (`create_issue_xml.py`) is 100% DETERMINISTIC and only:
- Takes the COMPLETE user input as the issue description
- Takes the issue prefix for debug logging
- Reads the files Claude specified
- Generates the project folder tree (folders only, no files)
- Creates XML format with the full context including `<debugging>` tag
- Copies to clipboard

The output is saved as `issue_description_[name].xml` and automatically copied to clipboard.

## Debug Logging Requirements
The XML output will include a `<debugging>` tag that specifies:
- When fixing the issue, add debug log prints to help diagnose if the fix doesn't work
- All debug prints must follow the structure: `[ISSUE_PREFIX] message`
- Example: `[ANIM_HANDOFF] Letter positions frozen at: [(350, 180), (375, 180), ...]`

## Examples
```
/summarize "text animation position jump when transitioning. I think we should store the final positions and pass them to the next animation"
# Claude decides issue prefix: "ANIM_HANDOFF"
# Claude passes FULL input including the solution suggestion
# Claude finds: utils/animations/text_behind_segment.py, utils/animations/word_dissolve.py, test_refactored_animations.py

/summarize "START text not rendering behind person"
# Claude decides issue prefix: "TEXT_BEHIND"
# Claude finds: specific animation files, not entire pipeline or lambda functions

/summarize "video segmentation not working - maybe it's the mask generation?"
# Claude decides issue prefix: "SEG_MASK"
# Claude passes FULL input including the hypothesis
# Claude finds: utils/video_segmentation/*.py, not all video-related files
```

## Implementation
Claude should:
1. Use Grep/Glob to find files based on specific keywords/modules in the issue
2. Be VERY selective - only include files directly related to the issue
3. Decide on appropriate issue prefix based on the problem domain
4. Run: `python create_issue_xml.py --issue "FULL USER INPUT HERE" --prefix "ISSUE_PREFIX" --files file1 file2 file3...`

## File Selection Guidelines
**BE EXTREMELY SELECTIVE:**
- **For test files**: Only include THE SPECIFIC test file that reproduces the issue (check which was actually run)
- **Avoid duplicates**: If multiple similar files exist (e.g., test_3d_motion.py, test_3d_composed.py), include ONLY the one actively being used
- **Core code only**: Include the minimal set of production code files directly involved in the issue
- **No peripheral files**: Exclude helper scripts, alternative implementations, or unused variants

**DO NOT include**: 
- Multiple test files for the same functionality
- Lambda functions, pipeline files, or deployment code unless specifically mentioned
- Alternative or backup implementations (_v2, _old, _backup files)
- Debug/verification scripts unless they contain the actual bug