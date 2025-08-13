import cv2
import numpy as np
import sys
import os

# Get input file from command line or use default
if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    input_file = 'cartoon-test/robot.png'

# Get output file from command line or use default
if len(sys.argv) > 2:
    output_file = sys.argv[2]
else:
    output_file = 'utils/contour_extraction/outline.png'

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Error: Input file {input_file} not found")
    sys.exit(1)

print(f"Processing: {input_file}")

img = cv2.imread(input_file, cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# מנקים רעש
blur = cv2.GaussianBlur(gray, (5,5), 0)

# קווי מתאר (שחק עם הספים)
edges = cv2.Canny(blur, threshold1=80, threshold2=160)

# הופכים ללבן על רקע שקוף
alpha = np.where(edges>0, 255, 0).astype(np.uint8)
rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
rgba[:,:,0:3] = 255                    # לבן לקו
rgba[:,:,3] = alpha                    # אלפא לפי קצה

cv2.imwrite(output_file, rgba)
print(f"Output saved to: {output_file}")