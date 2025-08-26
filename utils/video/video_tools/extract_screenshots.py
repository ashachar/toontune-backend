import cv2
import os

video_path = "hand_drawing_woman.mp4"
output_dir = "screenshots"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Total frames: {total_frames}, FPS: {fps}")

# Take 8 screenshots at uniform intervals
num_screenshots = 8
interval = total_frames // num_screenshots

for i in range(num_screenshots):
    frame_num = i * interval
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if ret:
        timestamp = frame_num / fps
        filename = f"{output_dir}/screenshot_{i+1}_at_{timestamp:.1f}s.png"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

cap.release()
print("Screenshots extracted!")