import cv2
import numpy as np
import os
import subprocess
import sys
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.spatial import distance_matrix
import math

def load_hand_image(hand_path):
    hand = cv2.imread(hand_path, cv2.IMREAD_UNCHANGED)
    
    if hand is not None and hand.shape[2] == 4:
        bgr = hand[:, :, :3]
        alpha = hand[:, :, 3]
        return bgr, alpha
    else:
        return hand, None

def overlay_hand(background, hand, hand_alpha, position):
    x, y = position
    h, w = hand.shape[:2]
    bg_h, bg_w = background.shape[:2]
    
    x = max(0, min(x, bg_w - w))
    y = max(0, min(y, bg_h - h))
    
    if x + w > bg_w:
        w = bg_w - x
    if y + h > bg_h:
        h = bg_h - y
    
    if w <= 0 or h <= 0:
        return background
    
    roi = background[y:y+h, x:x+w]
    hand_crop = hand[:h, :w]
    
    if hand_alpha is not None:
        alpha_crop = hand_alpha[:h, :w]
        alpha = alpha_crop.astype(float) / 255
        alpha = np.expand_dims(alpha, axis=2)
        
        blended = (1 - alpha) * roi + alpha * hand_crop
        background[y:y+h, x:x+w] = blended.astype(np.uint8)
    else:
        background[y:y+h, x:x+w] = hand_crop
    
    return background

def is_white_or_transparent(color, threshold=240):
    """Check if a color is white or very light"""
    return all(c > threshold for c in color[:3])

def segment_image_with_meanshift(image_path):
    """
    Segment the image using Mean Shift clustering - more robust than K-means
    """
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Create feature vectors for non-white pixels only
    features = []
    positions = []
    
    # Sample the image, but skip white/light pixels
    step = 3
    for y in range(0, height, step):
        for x in range(0, width, step):
            color = img[y, x]
            
            # Skip white or very light pixels
            if is_white_or_transparent(color, threshold=240):
                continue
            
            # Normalize position features
            pos_x = x / width
            pos_y = y / height
            
            # Get color features (normalized)
            b, g, r = color / 255.0
            
            # Create feature vector with strong spatial weighting
            feature = [pos_x * 4, pos_y * 4, r * 0.5, g * 0.5, b * 0.5]
            features.append(feature)
            positions.append((x, y))
    
    if len(features) < 100:
        print("Warning: Very few non-white pixels found")
        return None, img
    
    features = np.array(features)
    
    # Estimate bandwidth for Mean Shift
    bandwidth = estimate_bandwidth(features, quantile=0.15, n_samples=min(500, len(features)))
    
    # Perform Mean Shift clustering
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = ms.fit_predict(features)
    
    n_clusters = len(set(labels))
    print(f"Mean Shift found {n_clusters} clusters")
    
    # Create cluster map
    cluster_map = np.full((height, width), -1, dtype=np.int32)  # -1 for white/background
    
    # Assign cluster labels to pixels
    for (x, y), label in zip(positions, labels):
        # Fill in the region around the sampled point
        for dy in range(-step, step+1):
            for dx in range(-step, step+1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    # Only assign if pixel is not white
                    if not is_white_or_transparent(img[ny, nx], threshold=240):
                        cluster_map[ny, nx] = label
    
    # Fill in any remaining non-white pixels with nearest cluster
    for y in range(height):
        for x in range(width):
            if cluster_map[y, x] == -1 and not is_white_or_transparent(img[y, x], threshold=240):
                # Find nearest labeled pixel
                min_dist = float('inf')
                nearest_label = 0
                for dy in range(-10, 11):
                    for dx in range(-10, 11):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width and cluster_map[ny, nx] >= 0:
                            dist = dy*dy + dx*dx
                            if dist < min_dist:
                                min_dist = dist
                                nearest_label = cluster_map[ny, nx]
                cluster_map[y, x] = nearest_label
    
    return cluster_map, img

def count_cluster_pixels(cluster_map, cluster_id):
    """Count non-background pixels in a cluster"""
    return np.sum(cluster_map == cluster_id)

def get_cluster_info(cluster_map):
    """Get information about all clusters"""
    unique_labels = np.unique(cluster_map)
    cluster_info = []
    
    for label in unique_labels:
        if label == -1:  # Skip background
            continue
        
        pixel_count = count_cluster_pixels(cluster_map, label)
        if pixel_count > 50:  # Only include clusters with significant content
            points = np.where(cluster_map == label)
            center_y = np.mean(points[0])
            center_x = np.mean(points[1])
            cluster_info.append({
                'id': label,
                'pixel_count': pixel_count,
                'center': (center_x, center_y)
            })
    
    return cluster_info

def order_clusters_clockwise(cluster_info):
    """Order clusters in clockwise direction starting from top"""
    if not cluster_info:
        return []
    
    # Find average center
    avg_x = np.mean([c['center'][0] for c in cluster_info])
    avg_y = np.mean([c['center'][1] for c in cluster_info])
    img_center = (avg_x, avg_y)
    
    # Calculate angle for each cluster center relative to image center
    def get_angle(cluster):
        dx = cluster['center'][0] - img_center[0]
        dy = cluster['center'][1] - img_center[1]
        # Start from top (12 o'clock) and go clockwise
        angle = math.atan2(dx, -dy)
        return angle
    
    # Sort clusters by angle
    cluster_info.sort(key=get_angle)
    
    return cluster_info

def find_cluster_boundary(cluster_map, cluster_id):
    """Find boundary pixels of a cluster"""
    # Create binary mask for this cluster
    mask = (cluster_map == cluster_id).astype(np.uint8)
    
    # Find contours (boundaries)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return []
    
    # Get the largest contour (main boundary)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Extract boundary points
    boundary_points = []
    for point in main_contour:
        x, y = point[0]
        boundary_points.append((x, y))
    
    return boundary_points

def order_boundary_clockwise(boundary_points, center):
    """Order boundary points in clockwise direction from top"""
    if not boundary_points:
        return []
    
    def get_angle(point):
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        return math.atan2(dx, -dy)  # Clockwise from top
    
    # Sort points by angle
    sorted_points = sorted(boundary_points, key=get_angle)
    
    # Sample points to reduce density (every nth point)
    step = max(1, len(sorted_points) // 100)  # Keep ~100 points
    sampled_points = sorted_points[::step]
    
    return sampled_points

def offset_boundary_inward(boundary_points, center, offset_ratio=0.3):
    """Offset boundary points toward center by given ratio"""
    offset_points = []
    
    for point in boundary_points:
        # Calculate vector from point to center
        dx = center[0] - point[0]
        dy = center[1] - point[1]
        
        # Move point toward center by offset_ratio
        new_x = point[0] + dx * offset_ratio
        new_y = point[1] + dy * offset_ratio
        
        offset_points.append((int(new_x), int(new_y)))
    
    return offset_points

def smooth_trajectory(points, window_size=5):
    """Smooth trajectory using moving average"""
    if len(points) < window_size:
        return points
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(points)):
        start = max(0, i - half_window)
        end = min(len(points), i + half_window + 1)
        
        avg_x = np.mean([points[j][0] for j in range(start, end)])
        avg_y = np.mean([points[j][1] for j in range(start, end)])
        
        smoothed.append((int(avg_x), int(avg_y)))
    
    return smoothed

def map_pixels_to_trajectory(cluster_map, cluster_id, trajectory):
    """Map each pixel in cluster to its nearest trajectory point"""
    # Get all pixels in the cluster
    points = np.where(cluster_map == cluster_id)
    cluster_pixels = list(zip(points[1], points[0]))  # (x, y) format
    
    if not cluster_pixels or not trajectory:
        return {}
    
    # Create mapping: trajectory_index -> list of pixels
    pixel_mapping = {i: [] for i in range(len(trajectory))}
    
    # For each pixel, find nearest trajectory point
    for pixel in cluster_pixels:
        min_dist = float('inf')
        nearest_idx = 0
        
        for idx, traj_point in enumerate(trajectory):
            dist = (pixel[0] - traj_point[0])**2 + (pixel[1] - traj_point[1])**2
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        
        pixel_mapping[nearest_idx].append(pixel)
    
    return pixel_mapping

def create_edge_following_path(cluster_map, cluster_id, center):
    """Create a path that follows the cluster edge, offset inward"""
    # Find boundary pixels
    boundary = find_cluster_boundary(cluster_map, cluster_id)
    
    if not boundary:
        return []
    
    # Order boundary points clockwise
    ordered_boundary = order_boundary_clockwise(boundary, center)
    
    # Offset boundary inward by 30%
    offset_boundary = offset_boundary_inward(ordered_boundary, center, offset_ratio=0.3)
    
    # Smooth the trajectory
    smooth_path = smooth_trajectory(offset_boundary, window_size=5)
    
    # Ensure path is closed (connect end to beginning)
    if smooth_path and smooth_path[0] != smooth_path[-1]:
        # Add interpolated points to close the loop smoothly
        last = smooth_path[-1]
        first = smooth_path[0]
        num_interp = 5
        for i in range(1, num_interp):
            t = i / num_interp
            x = int(last[0] * (1 - t) + first[0] * t)
            y = int(last[1] * (1 - t) + first[1] * t)
            smooth_path.append((x, y))
    
    return smooth_path

def create_drawing_animation():
    woman_path = "images/woman.png"
    hand_path = "images/hand.png"
    output_video = "hand_drawing_woman.mp4"
    
    print("Segmenting woman image with Mean Shift clustering...")
    cluster_map, woman_img = segment_image_with_meanshift(woman_path)
    
    if cluster_map is None:
        print("Error: Failed to segment image")
        return None
    
    height, width = woman_img.shape[:2]
    
    print("Loading hand image...")
    hand_img, hand_alpha = load_hand_image(hand_path)
    
    hand_scale = 0.12
    new_width = int(hand_img.shape[1] * hand_scale)
    new_height = int(hand_img.shape[0] * hand_scale)
    hand_img = cv2.resize(hand_img, (new_width, new_height))
    if hand_alpha is not None:
        hand_alpha = cv2.resize(hand_alpha, (new_width, new_height))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    reveal_mask = np.zeros((height, width), dtype=np.uint8)
    
    pen_offset_x = new_width // 3
    pen_offset_y = new_height // 3
    
    print("Analyzing clusters and filtering out white/transparent regions...")
    cluster_info = get_cluster_info(cluster_map)
    
    print(f"Found {len(cluster_info)} valid clusters with content")
    
    # Calculate total pixels to draw
    total_pixels = sum(c['pixel_count'] for c in cluster_info)
    
    print("Ordering clusters in clockwise direction...")
    ordered_clusters = order_clusters_clockwise(cluster_info)
    
    # Print cluster information
    for i, cluster in enumerate(ordered_clusters):
        percentage = (cluster['pixel_count'] / total_pixels) * 100
        print(f"Cluster {i+1}: {cluster['pixel_count']} pixels ({percentage:.1f}% of content)")
    
    # Reduced total frames by 50% for even faster video
    total_frames = 90  # 3 seconds at 30fps for main drawing
    frames_used = 0
    
    print(f"Creating drawing animation with edge-following pattern (3 seconds)...")
    
    # Pre-calculate exact frame allocation for each cluster
    frame_allocation = []
    remaining_frames = total_frames
    
    for i, cluster in enumerate(ordered_clusters):
        if i == len(ordered_clusters) - 1:
            # Give all remaining frames to last cluster to avoid rounding issues
            frames = remaining_frames
        else:
            # Strictly proportional allocation
            frames = int((cluster['pixel_count'] / total_pixels) * total_frames)
            frames = max(5, frames)  # Minimum 5 frames per cluster
            remaining_frames -= frames
        frame_allocation.append(frames)
    
    print(f"Frame allocation: {frame_allocation}")
    
    for cluster_idx, cluster in enumerate(ordered_clusters):
        cluster_id = cluster['id']
        pixel_count = cluster['pixel_count']
        center = cluster['center']
        
        # Use pre-calculated frame allocation
        frames_for_cluster = frame_allocation[cluster_idx]
        
        print(f"Drawing cluster {cluster_idx + 1}/{len(ordered_clusters)} ({frames_for_cluster} frames)...")
        print(f"  Creating edge-following trajectory...")
        
        # Create edge-following path
        trajectory = create_edge_following_path(cluster_map, cluster_id, center)
        
        if not trajectory:
            print(f"  Warning: Could not create trajectory for cluster {cluster_idx + 1}")
            continue
        
        print(f"  Trajectory has {len(trajectory)} points")
        print(f"  Mapping pixels to trajectory points...")
        
        # Map all cluster pixels to nearest trajectory points
        pixel_mapping = map_pixels_to_trajectory(cluster_map, cluster_id, trajectory)
        
        # Calculate how many trajectory points to visit per frame
        points_per_frame = max(1, len(trajectory) // frames_for_cluster)
        
        trajectory_idx = 0
        cluster_frames = 0
        
        while trajectory_idx < len(trajectory) and cluster_frames < frames_for_cluster and frames_used < total_frames:
            frame = canvas.copy()
            
            # Process multiple trajectory points per frame
            for _ in range(points_per_frame):
                if trajectory_idx < len(trajectory):
                    # Get all pixels mapped to this trajectory point
                    pixels_to_reveal = pixel_mapping.get(trajectory_idx, [])
                    
                    # Reveal these pixels with larger brush to catch more area
                    for px, py in pixels_to_reveal:
                        cv2.circle(reveal_mask, (px, py), 12, 255, -1)
                    
                    # Also reveal pixels near the trajectory point itself for better coverage
                    if trajectory_idx < len(trajectory):
                        traj_x, traj_y = trajectory[trajectory_idx]
                        cv2.circle(reveal_mask, (traj_x, traj_y), 20, 255, -1)
                    
                    trajectory_idx += 1
            
            # Apply revealed portions of the woman image
            mask_3channel = cv2.cvtColor(reveal_mask, cv2.COLOR_GRAY2BGR) / 255.0
            frame = (frame * (1 - mask_3channel) + woman_img * mask_3channel).astype(np.uint8)
            
            # Add hand at current trajectory position
            if trajectory_idx > 0 and trajectory_idx <= len(trajectory):
                hand_pos = trajectory[min(trajectory_idx - 1, len(trajectory) - 1)]
                hand_x = hand_pos[0] - pen_offset_x
                hand_y = hand_pos[1] - pen_offset_y
                frame = overlay_hand(frame, hand_img, hand_alpha, (hand_x, hand_y))
            
            video_writer.write(frame)
            frames_used += 1
            cluster_frames += 1
            
            if frames_used % 30 == 0:
                print(f"Progress: {frames_used}/{total_frames} frames")
    
    # Check for any missing pixels and do final touch-ups
    print("Checking for missed pixels...")
    
    # Find pixels that should be drawn but weren't
    missed_pixels = []
    for y in range(height):
        for x in range(width):
            if cluster_map[y, x] >= 0 and reveal_mask[y, x] == 0:
                missed_pixels.append((x, y))
    
    if missed_pixels:
        print(f"Found {len(missed_pixels)} missed pixels, adding final touches...")
        
        # Group missed pixels by proximity
        missed_groups = []
        used = set()
        
        for px, py in missed_pixels:
            if (px, py) in used:
                continue
            
            # Find nearby missed pixels
            group = [(px, py)]
            used.add((px, py))
            
            for qx, qy in missed_pixels:
                if (qx, qy) not in used:
                    dist = (px - qx)**2 + (py - qy)**2
                    if dist < 400:  # Within 20 pixels
                        group.append((qx, qy))
                        used.add((qx, qy))
            
            if group:
                # Calculate center of group
                center_x = int(np.mean([p[0] for p in group]))
                center_y = int(np.mean([p[1] for p in group]))
                missed_groups.append((center_x, center_y, group))
        
        # Quick animation for touch-ups (0.5 seconds)
        touch_up_frames = 15
        groups_per_frame = max(1, len(missed_groups) // touch_up_frames)
        
        group_idx = 0
        for frame_num in range(touch_up_frames):
            frame = canvas.copy()
            
            # Process some groups in this frame
            for _ in range(groups_per_frame):
                if group_idx < len(missed_groups):
                    center_x, center_y, group = missed_groups[group_idx]
                    
                    # Reveal all pixels in this group
                    for px, py in group:
                        cv2.circle(reveal_mask, (px, py), 8, 255, -1)
                    
                    group_idx += 1
            
            # Apply revealed portions
            mask_3channel = cv2.cvtColor(reveal_mask, cv2.COLOR_GRAY2BGR) / 255.0
            frame = (frame * (1 - mask_3channel) + woman_img * mask_3channel).astype(np.uint8)
            
            # Add hand at current touch-up location
            if group_idx > 0 and group_idx <= len(missed_groups):
                hand_x = missed_groups[min(group_idx - 1, len(missed_groups) - 1)][0] - pen_offset_x
                hand_y = missed_groups[min(group_idx - 1, len(missed_groups) - 1)][1] - pen_offset_y
                frame = overlay_hand(frame, hand_img, hand_alpha, (hand_x, hand_y))
            
            video_writer.write(frame)
        
        print(f"Added {touch_up_frames} frames for final touches")
    
    # Ensure the entire image is revealed
    reveal_mask[:, :] = 255
    final_frame = woman_img.copy()
    
    # Add some frames showing the completed image
    for _ in range(30):  # Reduced from 60 to 30 for faster video
        video_writer.write(final_frame)
    
    video_writer.release()
    cv2.destroyAllWindows()
    
    total_duration = (frames_used + (touch_up_frames if missed_pixels else 0) + 30) / fps
    print(f"Video saved as {output_video} (duration: {total_duration:.1f} seconds)")
    return output_video

def open_video(video_path):
    if sys.platform == "win32":
        os.startfile(video_path)
    elif sys.platform == "darwin":
        subprocess.run(["open", video_path])
    else:
        subprocess.run(["xdg-open", video_path])

if __name__ == "__main__":
    video_file = create_drawing_animation()
    if video_file:
        print(f"Opening video: {video_file}")
        open_video(video_file)