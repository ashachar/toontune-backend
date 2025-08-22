import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session
import matplotlib.pyplot as plt

# Initialize rembg session
session = new_session('u2net')

# Load a frame from the video
video_path = "lambda_output.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if ret:
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Generate mask with rembg
    output = remove(pil_image, session=session)
    output_np = np.array(output)
    
    # Extract alpha channel as mask
    if output_np.shape[2] == 4:
        mask = output_np[:, :, 3]
    else:
        gray = cv2.cvtColor(output_np, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Save results
    cv2.imwrite("test_frame.png", frame)
    cv2.imwrite("test_mask.png", mask)
    
    # Create composite to visualize
    composite = np.zeros_like(frame)
    composite[:, :, 1] = mask  # Green channel for mask
    composite[:, :, 2] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Red channel for original
    cv2.imwrite("test_composite.png", composite)
    
    print(f"Frame shape: {frame.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {np.unique(mask)}")
    print("Saved: test_frame.png, test_mask.png, test_composite.png")
else:
    print("Failed to read video")