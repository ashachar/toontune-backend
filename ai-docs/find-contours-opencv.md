import cv2
import numpy as np

img = cv2.imread('input.png', cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# מנקים רעש
blur = cv2.GaussianBlur(gray, (5,5), 0)

# קווי מתאר (שחק עם הספים)
edges = cv2.Canny(blur, threshold1=80, threshold2=160)

# הופכים לשחור על רקע שקוף
alpha = np.where(edges>0, 255, 0).astype(np.uint8)
rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
rgba[:,:,0:3] = 0                      # שחור לקו
rgba[:,:,3] = alpha                    # אלפא לפי קצה

cv2.imwrite('outline.png', rgba)
