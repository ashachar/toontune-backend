# Step-by-Step Debugging Guide for Behind Text Issue

## Quick Start
```bash
python debug_behind_text.py
```

## Debugging in Cursor (Recommended)

### 1. Set Up Breakpoint
1. Open `debug_behind_text.py` in Cursor
2. Go to **line 87** (the `breakpoint()` line)
3. Click in the gutter (left of line number) to add a red breakpoint dot
4. Or just leave the `breakpoint()` statement - Python will stop there

### 2. Start Debugging
**Method A: Using F5**
1. Press `F5` 
2. Select "Python File" when prompted
3. The script will run and stop at the breakpoint

**Method B: Using Debug Panel**
1. Click the "Run and Debug" icon in the left sidebar (play button with bug)
2. Click "Run and Debug" button
3. Select "Python File"

**Method C: Using Command Palette**
1. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
2. Type "Debug: Start Debugging"
3. Select "Python File"

### 3. Debug Controls
When the debugger stops at frame 85 (3.4 seconds):

**Navigation:**
- `F10` - Step Over (execute current line)
- `F11` - Step Into (go into function calls)
- `Shift+F11` - Step Out (exit current function)
- `F5` - Continue (run until next breakpoint)
- `Shift+F5` - Stop debugging

**Inspect Variables:**
- Hover over any variable to see its value
- Use the "Variables" panel in debug sidebar to explore:
  - `mask_frame` - The green screen frame
  - `bg_mask` - Background mask (1.0=show text, 0.0=hide)
  - `text_alpha` - Original text opacity
  - `final_alpha` - Text opacity after masking
  - `visible_pixels` - Count of visible pixels

### 4. Debug Console
While stopped at breakpoint, use the Debug Console to:
```python
# Check specific pixels
bg_mask[0, 0]  # Check top-left pixel

# Visualize arrays
import matplotlib.pyplot as plt
plt.imshow(bg_mask, cmap='gray')
plt.show()

# Check statistics
np.mean(bg_mask)
np.sum(final_alpha > 0.1)

# Inspect mask at specific coordinates
mask_region[30:40, 50:60]
```

### 5. Key Variables to Inspect

**At the breakpoint, examine these critical variables:**

1. **`mask_frame`** - Shape should be (720, 1280, 3)
   - This is the green screen video frame
   - Green pixels = background (where text shows)
   - Person pixels = foreground (where text hides)

2. **`bg_mask`** - Shape should match text size
   - Values: 0.0 (hide text) to 1.0 (show text)
   - Should be mostly 0.0 in head area
   - Should be 1.0 in background areas

3. **`final_alpha`** - Text visibility after masking
   - Result of `text_alpha * bg_mask`
   - Values > 0.1 will be visible
   - Should be near 0 where head is

4. **`visibility_percent`** - Should be low (~3-5%)
   - Percentage of text that will be visible
   - If too high, text leaks through
   - If 0%, text is completely hidden

### 6. Common Issues to Check

**Issue: Text showing on face**
```python
# In debug console, check if mask has holes:
problem_area = bg_mask[0:10, :]  # Top rows
np.sum(problem_area > 0.5)  # Should be very low
```

**Issue: Text not visible on background**
```python
# Check text color vs background
TEXT_COLOR  # Should be dark (50, 50, 100)
frame[30, 100]  # Check background color at that position
```

**Issue: Mask not aligned**
```python
# Check if mask and video frames match
mask_frame.shape  # Should be (720, 1280, 3)
frame.shape  # Should be same
```

### 7. Visualization (Optional)
Uncomment lines 116-127 in the script to see visual plots of:
- Original frame
- Green screen mask
- Background mask
- Text alpha before/after masking

### 8. Modify and Test
While debugging, you can modify values in the Debug Console:
```python
# Try different tolerance
TOLERANCE = 30  # Tighter tolerance
# Recalculate mask
distance = np.sqrt(np.sum(diff * diff, axis=2))
is_green = (distance < TOLERANCE)
new_bg_mask = is_green.astype(np.float32)
```

## Output Files Created
- The script reads from:
  - `outputs/ai_math1_word_level_h264.mp4` (rendered video)
  - `uploads/assets/videos/ai_math1/ai_math1_rvm_mask.mp4` (green screen mask)

## Tips
1. Use the Variables panel to expand numpy arrays and see their shape/dtype
2. Right-click variables to "Add to Watch" for persistent monitoring
3. Use conditional breakpoints: Right-click breakpoint → "Edit Breakpoint" → Add condition
4. The Debug Console supports full Python - import any libraries you need

## Next Steps
After identifying the issue:
1. Note which variables have unexpected values
2. Check the corresponding code in `pipelines/word_level_pipeline/rendering.py`
3. Make fixes and re-run the pipeline
4. Use this debug script again to verify the fix