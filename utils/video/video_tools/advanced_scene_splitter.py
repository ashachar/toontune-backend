#!/usr/bin/env python3
"""
Video Scene Splitter Utility

Automatically detects scene boundaries in videos based on:
- Color histogram changes (HSV color space analysis)
- Edge detection changes (structural changes)
- Blob detection (number and position of objects)
- Audio track analysis (volume, silence, frequency changes)

Uses clustering to group similar frames and identify scene boundaries.
"""

import argparse
import subprocess
import json
import sys
import os
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy import signal
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class SceneSplitter:
    """Video scene splitter with configurable thresholds"""
    
    def __init__(self, 
                 sensitivity=0.5,
                 min_scene_duration=1.0,
                 sample_rate=5,
                 use_audio=True,
                 verbose=True,
                 export_analysis=True):
        """
        Initialize SceneSplitter
        
        Args:
            sensitivity: Scene change detection sensitivity (0-1, higher = more sensitive)
            min_scene_duration: Minimum scene duration in seconds
            sample_rate: Process every Nth frame for efficiency
            use_audio: Whether to include audio analysis
            verbose: Print progress messages
            export_analysis: Export frame-by-frame analysis to Excel
        """
        self.sensitivity = sensitivity
        self.min_scene_duration = min_scene_duration
        self.sample_rate = sample_rate
        self.use_audio = use_audio
        self.verbose = verbose
        self.export_analysis = export_analysis
        
        # Threshold mappings based on sensitivity
        self.color_threshold = 0.3 + (0.4 * (1 - sensitivity))  # 0.3-0.7
        self.edge_threshold = 0.25 + (0.35 * (1 - sensitivity))  # 0.25-0.6
        self.blob_threshold = 0.4 + (0.3 * (1 - sensitivity))    # 0.4-0.7
        self.audio_threshold = 0.35 + (0.35 * (1 - sensitivity)) # 0.35-0.7
        
        # Initialize blob detector
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 10
        params.maxThreshold = 200
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 10000
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        self.blob_detector = cv2.SimpleBlobDetector_create(params)
        
        # Initialize face/person detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        
        # Store detailed analysis data
        self.frame_analysis_data = []
    
    def _log(self, message):
        """Print log message if verbose"""
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def get_video_info(self, video_path):
        """Get video metadata using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,duration,nb_frames',
                '-of', 'json',
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                info = json.loads(result.stdout)
                stream = info['streams'][0]
                
                # Parse frame rate
                fps_str = stream.get('r_frame_rate', '25/1')
                fps_parts = fps_str.split('/')
                fps = float(fps_parts[0]) / float(fps_parts[1])
                
                return {
                    'width': int(stream.get('width', 0)),
                    'height': int(stream.get('height', 0)),
                    'fps': fps,
                    'duration': float(stream.get('duration', 0)),
                    'total_frames': int(stream.get('nb_frames', 0))
                }
        except Exception as e:
            self._log(f"Error getting video info: {e}")
        
        # Fallback to OpenCV
        cap = cv2.VideoCapture(str(video_path))
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        info['duration'] = info['total_frames'] / info['fps'] if info['fps'] > 0 else 0
        cap.release()
        return info
    
    def extract_color_histogram(self, frame):
        """Extract HSV color histogram features"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])
        
        # Normalize histograms
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Concatenate and create feature vector
        hist_features = np.concatenate([hist_h, hist_s, hist_v])
        
        return hist_features
    
    def extract_edge_features(self, frame):
        """Extract edge detection features using Canny"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate edge statistics
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_density = edge_pixels / total_pixels
        
        # Divide image into grid for spatial distribution
        h, w = edges.shape
        grid_h, grid_w = 4, 4
        cell_h, cell_w = h // grid_h, w // grid_w
        
        edge_distribution = []
        for i in range(grid_h):
            for j in range(grid_w):
                cell = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_density = np.sum(cell > 0) / (cell_h * cell_w)
                edge_distribution.append(cell_density)
        
        # Combine features
        edge_features = np.array([edge_density] + edge_distribution)
        
        return edge_features, edges
    
    def extract_blob_features(self, frame):
        """Extract blob detection features for object counting"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect blobs
        keypoints = self.blob_detector.detect(gray)
        
        # Extract blob statistics
        num_blobs = len(keypoints)
        
        # Calculate spatial distribution
        h, w = gray.shape
        blob_positions = []
        blob_sizes = []
        
        for kp in keypoints:
            # Normalize position
            norm_x = kp.pt[0] / w
            norm_y = kp.pt[1] / h
            blob_positions.extend([norm_x, norm_y])
            blob_sizes.append(kp.size)
        
        # Create fixed-size feature vector
        max_blobs = 10
        blob_features = [num_blobs / max_blobs]  # Normalized count
        
        # Add position features (pad if needed)
        position_features = blob_positions[:max_blobs*2]
        position_features.extend([0] * (max_blobs*2 - len(position_features)))
        blob_features.extend(position_features)
        
        # Add size features
        size_features = blob_sizes[:max_blobs]
        size_features.extend([0] * (max_blobs - len(size_features)))
        if size_features:
            max_size = max(size_features) if max(size_features) > 0 else 1
            size_features = [s / max_size for s in size_features]
        blob_features.extend(size_features)
        
        return np.array(blob_features)
    
    def extract_audio_features(self, video_path, start_time, duration):
        """Extract audio features for a segment"""
        try:
            # Extract raw audio data using ffmpeg
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-vn',  # No video
                '-ar', '22050',  # Sample rate
                '-ac', '1',  # Mono
                '-f', 's16le',  # Format
                '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                return None
            
            # Convert to numpy array
            audio_data = np.frombuffer(result.stdout, dtype=np.int16)
            if len(audio_data) == 0:
                return None
            
            # Normalize
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Calculate features
            features = []
            
            # 1. Energy/Volume
            energy = np.sqrt(np.mean(audio_data ** 2))
            features.append(energy)
            
            # 2. Zero crossing rate (indicates frequency content)
            zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0) / len(audio_data)
            features.append(zero_crossings)
            
            # 3. Silence ratio
            silence_threshold = 0.01
            silence_ratio = np.sum(np.abs(audio_data) < silence_threshold) / len(audio_data)
            features.append(silence_ratio)
            
            # 4. Dynamic range
            if len(audio_data) > 0:
                percentile_90 = np.percentile(np.abs(audio_data), 90)
                percentile_10 = np.percentile(np.abs(audio_data), 10)
                dynamic_range = percentile_90 - percentile_10
            else:
                dynamic_range = 0
            features.append(dynamic_range)
            
            # 5. Spectral centroid (brightness)
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_data), 1/22050)
            
            if np.sum(magnitude) > 0:
                spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                spectral_centroid = spectral_centroid / 11025  # Normalize by Nyquist
            else:
                spectral_centroid = 0
            features.append(spectral_centroid)
            
            return np.array(features)
            
        except Exception as e:
            self._log(f"Audio extraction error: {e}")
            return None
    
    def detect_people(self, frame):
        """Detect faces and people in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        num_faces = len(faces)
        
        # Detect bodies (full person)
        bodies = self.body_cascade.detectMultiScale(gray, 1.1, 3)
        num_bodies = len(bodies)
        
        # Use maximum of faces and bodies as person count
        # (faces are more reliable but bodies catch people from behind)
        num_people = max(num_faces, num_bodies)
        
        # Calculate face positions and sizes
        face_positions = []
        face_sizes = []
        for (x, y, w, h) in faces:
            # Normalize position to frame size
            center_x = (x + w/2) / frame.shape[1]
            center_y = (y + h/2) / frame.shape[0]
            face_positions.extend([center_x, center_y])
            face_sizes.append((w * h) / (frame.shape[0] * frame.shape[1]))
        
        return {
            'num_people': num_people,
            'num_faces': num_faces,
            'num_bodies': num_bodies,
            'face_positions': face_positions[:10],  # Limit to 5 faces (10 coords)
            'face_sizes': face_sizes[:5],
            'largest_face_size': max(face_sizes) if face_sizes else 0,
            'avg_face_size': np.mean(face_sizes) if face_sizes else 0
        }
    
    def extract_features(self, video_path):
        """Extract all features from video frames"""
        cap = cv2.VideoCapture(str(video_path))
        
        video_info = self.get_video_info(video_path)
        fps = video_info['fps']
        total_frames = video_info['total_frames']
        
        self._log(f"Processing video: {video_info['width']}x{video_info['height']}, "
                 f"{fps:.2f} fps, {total_frames} frames")
        
        frame_features = []
        frame_indices = []
        frame_count = 0
        
        # Clear analysis data
        self.frame_analysis_data = []
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames for efficiency
            if frame_count % self.sample_rate == 0:
                timestamp = frame_count / fps
                
                # Resize for faster processing
                scale = 0.5
                height, width = frame.shape[:2]
                new_height = int(height * scale)
                new_width = int(width * scale)
                frame_resized = cv2.resize(frame, (new_width, new_height))
                
                # Extract visual features
                color_hist = self.extract_color_histogram(frame_resized)
                edge_features, edges = self.extract_edge_features(frame_resized)
                blob_features = self.extract_blob_features(frame_resized)
                
                # Detect people
                people_info = self.detect_people(frame)
                
                # Calculate additional metrics
                # Dominant colors (top 3 HSV values)
                hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
                h_mean = np.mean(hsv[:,:,0])
                s_mean = np.mean(hsv[:,:,1])
                v_mean = np.mean(hsv[:,:,2])
                
                # Motion/blur detection (Laplacian variance)
                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # Store detailed frame analysis
                frame_data = {
                    'frame_num': frame_count,
                    'timestamp': round(timestamp, 2),
                    'num_people': people_info['num_people'],
                    'num_faces': people_info['num_faces'],
                    'num_bodies': people_info['num_bodies'],
                    'num_blobs': int(blob_features[0] * 10),  # Denormalize
                    'edge_density': round(float(edge_features[0]), 3),
                    'h_mean': round(h_mean, 1),
                    's_mean': round(s_mean, 1),
                    'v_mean': round(v_mean, 1),
                    'brightness': round(v_mean / 255.0, 3),
                    'blur_measure': round(laplacian_var, 2),
                    'largest_face_size': round(people_info['largest_face_size'], 4),
                    'avg_face_size': round(people_info['avg_face_size'], 4)
                }
                
                self.frame_analysis_data.append(frame_data)
                
                # Combine features (including people detection)
                people_features = np.array([
                    people_info['num_people'] / 10.0,  # Normalize
                    people_info['num_faces'] / 10.0,
                    people_info['num_bodies'] / 10.0,
                    people_info['largest_face_size'],
                    people_info['avg_face_size']
                ])
                
                combined_features = np.concatenate([
                    color_hist,
                    edge_features,
                    blob_features,
                    people_features
                ])
                
                frame_features.append(combined_features)
                frame_indices.append(frame_count)
                
                if len(frame_features) % 50 == 0:
                    self._log(f"Processed {len(frame_features)} sampled frames...")
            
            frame_count += 1
        
        cap.release()
        
        # Extract audio features if enabled
        audio_features = []
        if self.use_audio:
            self._log("Extracting audio features...")
            
            # Sample audio at regular intervals
            audio_sample_duration = 0.5  # seconds
            audio_sample_interval = self.sample_rate / fps
            
            for i, frame_idx in enumerate(frame_indices):
                timestamp = frame_idx / fps
                audio_feat = self.extract_audio_features(
                    video_path, timestamp, audio_sample_duration
                )
                
                if audio_feat is not None:
                    audio_features.append(audio_feat)
                else:
                    # Use zeros if audio extraction fails
                    audio_features.append(np.zeros(5))
        
        return {
            'frame_features': np.array(frame_features),
            'audio_features': np.array(audio_features) if audio_features else None,
            'frame_indices': frame_indices,
            'fps': fps,
            'total_frames': total_frames
        }
    
    def export_frame_analysis(self, video_path):
        """Export detailed frame analysis to Excel/CSV"""
        if not self.frame_analysis_data:
            self._log("No frame analysis data to export")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(self.frame_analysis_data)
        
        # Add change detection columns
        df['people_change'] = df['num_people'].diff().abs()
        df['faces_change'] = df['num_faces'].diff().abs()
        df['brightness_change'] = df['brightness'].diff().abs()
        df['blur_change'] = df['blur_measure'].diff().abs()
        
        # Mark potential scene changes
        df['potential_scene'] = False
        
        # Detect significant changes in people count
        people_threshold = 2  # Change of 2+ people
        df.loc[df['people_change'] >= people_threshold, 'potential_scene'] = True
        
        # Detect significant brightness changes
        brightness_threshold = 0.3  # 30% brightness change
        df.loc[df['brightness_change'] >= brightness_threshold, 'potential_scene'] = True
        
        # Detect significant blur changes (focus changes)
        blur_std = df['blur_change'].std()
        blur_threshold = 2 * blur_std if blur_std > 0 else 100
        df.loc[df['blur_change'] >= blur_threshold, 'potential_scene'] = True
        
        # Save to Excel
        video_name = Path(video_path).stem
        output_dir = Path(video_path).parent
        excel_path = output_dir / f"{video_name}_frame_analysis.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Frame Analysis', index=False)
            
            # Add summary sheet
            summary_data = {
                'Metric': [
                    'Total Frames Analyzed',
                    'Average People per Frame',
                    'Max People in Frame',
                    'Min People in Frame',
                    'Average Brightness',
                    'Brightness Std Dev',
                    'Average Blur',
                    'Potential Scene Changes'
                ],
                'Value': [
                    len(df),
                    round(df['num_people'].mean(), 2),
                    df['num_people'].max(),
                    df['num_people'].min(),
                    round(df['brightness'].mean(), 3),
                    round(df['brightness'].std(), 3),
                    round(df['blur_measure'].mean(), 2),
                    df['potential_scene'].sum()
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Also save as CSV for compatibility
        csv_path = output_dir / f"{video_name}_frame_analysis.csv"
        df.to_csv(csv_path, index=False)
        
        self._log(f"Frame analysis exported to:")
        self._log(f"  Excel: {excel_path}")
        self._log(f"  CSV: {csv_path}")
        
        return excel_path
    
    def detect_scene_changes(self, features_dict):
        """Detect scene changes using feature differences and clustering"""
        frame_features = features_dict['frame_features']
        audio_features = features_dict['audio_features']
        frame_indices = features_dict['frame_indices']
        fps = features_dict['fps']
        
        n_frames = len(frame_features)
        
        # Calculate frame-to-frame differences
        visual_diffs = []
        people_diffs = []
        
        # Track people count directly from analysis data for better detection
        people_counts = []
        if self.frame_analysis_data:
            for data in self.frame_analysis_data:
                people_counts.append(data['num_people'])
        
        for i in range(1, n_frames):
            # Use multiple distance metrics
            hist_diff = cosine(frame_features[i][:170], frame_features[i-1][:170])  # Color histogram
            edge_diff = cosine(frame_features[i][170:187], frame_features[i-1][170:187])  # Edges
            blob_diff = np.linalg.norm(frame_features[i][187:218] - frame_features[i-1][187:218])  # Blobs
            
            # People detection differences (very important for scene changes)
            people_features_start = 218  # Adjust based on actual feature vector
            people_diff = np.linalg.norm(frame_features[i][people_features_start:people_features_start+5] - 
                                        frame_features[i-1][people_features_start:people_features_start+5])
            people_diffs.append(people_diff)
            
            # Check for significant people count change
            if people_counts and i < len(people_counts):
                actual_people_change = abs(people_counts[i] - people_counts[i-1])
                if actual_people_change >= 1:  # Person enters or leaves
                    people_diff = max(people_diff, 1.0)  # Force high difference score
            
            # Weighted combination (increased weight for people changes)
            visual_diff = (hist_diff * 0.25 + edge_diff * 0.15 + blob_diff * 0.15 + people_diff * 0.45)
            visual_diffs.append(visual_diff)
        
        visual_diffs = np.array(visual_diffs)
        people_diffs = np.array(people_diffs)
        
        # Add audio differences if available
        if audio_features is not None and len(audio_features) > 1:
            audio_diffs = []
            for i in range(1, len(audio_features)):
                audio_diff = np.linalg.norm(audio_features[i] - audio_features[i-1])
                audio_diffs.append(audio_diff)
            
            audio_diffs = np.array(audio_diffs)
            
            # Normalize and combine
            if len(visual_diffs) > 0 and len(audio_diffs) > 0:
                visual_diffs_norm = (visual_diffs - np.mean(visual_diffs)) / (np.std(visual_diffs) + 1e-6)
                audio_diffs_norm = (audio_diffs - np.mean(audio_diffs)) / (np.std(audio_diffs) + 1e-6)
                
                # Combine with weights
                combined_diffs = visual_diffs_norm * 0.7 + audio_diffs_norm * 0.3
            else:
                combined_diffs = visual_diffs
        else:
            combined_diffs = visual_diffs
        
        # Apply temporal smoothing
        if len(combined_diffs) > 5:
            kernel_size = 5
            kernel = np.ones(kernel_size) / kernel_size
            combined_diffs = signal.convolve(combined_diffs, kernel, mode='same')
        
        # Find peaks (scene changes)
        # Dynamic threshold based on statistics
        mean_diff = np.mean(combined_diffs)
        std_diff = np.std(combined_diffs)
        
        # Adjust threshold based on sensitivity
        threshold = mean_diff + (2.0 - 1.5 * self.sensitivity) * std_diff
        
        # Also detect based on people changes directly
        scene_boundaries = [0]  # Start with first frame
        
        # Method 1: Statistical peaks
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(combined_diffs, 
                                      height=threshold,
                                      distance=int(fps * self.min_scene_duration / self.sample_rate))
        
        for peak_idx in peaks:
            frame_idx = frame_indices[peak_idx + 1]  # +1 because diffs are offset
            scene_boundaries.append(frame_idx)
        
        # Method 2: Direct people count changes (more aggressive)
        if people_counts and self.sensitivity > 0.6:
            for i in range(1, len(people_counts)):
                if abs(people_counts[i] - people_counts[i-1]) >= 1:
                    # Person entered or left the frame
                    frame_idx = frame_indices[i]
                    # Only add if not too close to existing boundary
                    min_distance = fps * self.min_scene_duration
                    is_far_enough = all(abs(frame_idx - b) > min_distance for b in scene_boundaries)
                    if is_far_enough:
                        scene_boundaries.append(frame_idx)
        
        # Sort and remove duplicates
        scene_boundaries = sorted(list(set(scene_boundaries)))
        
        # Add last frame if not already included
        if len(scene_boundaries) == 0 or scene_boundaries[-1] != features_dict['total_frames']:
            scene_boundaries.append(features_dict['total_frames'])
        
        self._log(f"Detected {len(scene_boundaries)-1} potential scene boundaries")
        
        return scene_boundaries
    
    def cluster_frames(self, features_dict, scene_boundaries):
        """Refine scene boundaries using clustering"""
        frame_features = features_dict['frame_features']
        fps = features_dict['fps']
        
        # If we have too many scenes, try clustering to merge similar ones
        if len(scene_boundaries) > 2:
            # Create scene-level features
            scene_features = []
            for i in range(len(scene_boundaries) - 1):
                start_idx = scene_boundaries[i]
                end_idx = scene_boundaries[i + 1]
                
                # Find corresponding feature indices
                start_feat_idx = 0
                end_feat_idx = len(frame_features) - 1
                
                for j, frame_idx in enumerate(features_dict['frame_indices']):
                    if frame_idx >= start_idx and start_feat_idx == 0:
                        start_feat_idx = j
                    if frame_idx >= end_idx:
                        end_feat_idx = j
                        break
                
                # Average features for this scene
                if end_feat_idx > start_feat_idx:
                    scene_feat = np.mean(frame_features[start_feat_idx:end_feat_idx], axis=0)
                else:
                    scene_feat = frame_features[start_feat_idx]
                
                scene_features.append(scene_feat)
            
            scene_features = np.array(scene_features)
            
            # Normalize features
            scaler = StandardScaler()
            scene_features_norm = scaler.fit_transform(scene_features)
            
            # Try clustering to find optimal number of scenes
            max_clusters = min(len(scene_features), 20)
            min_clusters = max(2, int(len(scene_features) * 0.3))
            
            # Use agglomerative clustering for temporal coherence
            best_labels = None
            best_score = -1
            
            for n_clusters in range(min_clusters, max_clusters + 1):
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='ward'
                )
                labels = clusterer.fit_predict(scene_features_norm)
                
                # Check if clustering respects temporal order (mostly)
                label_changes = np.sum(np.diff(labels) != 0)
                if label_changes == n_clusters - 1:
                    # Perfect temporal clustering
                    best_labels = labels
                    break
            
            # Merge scenes with same cluster label
            if best_labels is not None:
                new_boundaries = [scene_boundaries[0]]
                for i in range(1, len(best_labels)):
                    if best_labels[i] != best_labels[i-1]:
                        new_boundaries.append(scene_boundaries[i])
                
                if new_boundaries[-1] != scene_boundaries[-1]:
                    new_boundaries.append(scene_boundaries[-1])
                
                return new_boundaries
        
        return scene_boundaries
    
    def split_scenes(self, video_path, save_clips=True):
        """Main method to split video into scenes
        
        Args:
            video_path: Path to input video
            save_clips: Whether to save individual scene video clips
        
        Returns:
            List of scene dictionaries with timestamps and file paths
        """
        self._log(f"Starting scene detection for: {video_path}")
        
        # Extract features
        features_dict = self.extract_features(video_path)
        
        # Export frame analysis if enabled
        if self.export_analysis and self.frame_analysis_data:
            excel_path = self.export_frame_analysis(video_path)
            if excel_path:
                self._log(f"✓ Frame-by-frame analysis exported to Excel")
        
        # Detect scene changes
        self._log("Detecting scene boundaries...")
        scene_boundaries = self.detect_scene_changes(features_dict)
        
        # Refine with clustering (skip if high sensitivity and many boundaries detected)
        if self.sensitivity < 0.75 or len(scene_boundaries) <= 3:
            self._log("Refining with clustering...")
            scene_boundaries = self.cluster_frames(features_dict, scene_boundaries)
        else:
            self._log(f"Keeping {len(scene_boundaries)-1} detected boundaries (high sensitivity mode)")
        
        # Convert to timestamp format
        fps = features_dict['fps']
        scenes = []
        
        self._log(f"Processing {len(scene_boundaries)-1} scene boundaries...")
        
        for i in range(len(scene_boundaries) - 1):
            start_frame = scene_boundaries[i]
            end_frame = scene_boundaries[i + 1]
            
            scene = {
                'start': start_frame / fps,
                'end': end_frame / fps,
                'duration': (end_frame - start_frame) / fps,
                'frame_start': int(start_frame),
                'frame_end': int(end_frame)
            }
            
            # Only include scenes longer than minimum duration
            if scene['duration'] >= self.min_scene_duration:
                scenes.append(scene)
                self._log(f"  Scene {len(scenes)}: {scene['start']:.1f}s - {scene['end']:.1f}s ({scene['duration']:.1f}s)")
            else:
                self._log(f"  Skipped short scene: {scene['start']:.1f}s - {scene['end']:.1f}s ({scene['duration']:.1f}s)")
        
        # Don't merge if we have high sensitivity - user wants the splits!
        if self.sensitivity >= 0.8:
            merged_scenes = scenes
            self._log(f"Keeping all {len(merged_scenes)} scenes (high sensitivity mode)")
        else:
            # Only merge if there's a gap (scenes were filtered)
            merged_scenes = []
            for scene in scenes:
                if merged_scenes and merged_scenes[-1]['end'] < scene['start'] - 0.5:
                    # There's a gap, keep separate
                    merged_scenes.append(scene)
                elif merged_scenes and merged_scenes[-1]['end'] == scene['start']:
                    # Adjacent due to filtering, merge
                    self._log(f"  Merging filtered scenes at {scene['start']:.1f}s")
                    merged_scenes[-1]['end'] = scene['end']
                    merged_scenes[-1]['duration'] = merged_scenes[-1]['end'] - merged_scenes[-1]['start']
                    merged_scenes[-1]['frame_end'] = scene['frame_end']
                else:
                    merged_scenes.append(scene)
            
            self._log(f"Final: {len(merged_scenes)} scenes after selective merging")
        
        # Save individual scene clips if requested
        if save_clips and len(merged_scenes) > 0:
            output_dir = self.save_scene_clips(video_path, merged_scenes)
            for i, scene in enumerate(merged_scenes):
                scene['clip_path'] = str(output_dir / f"scene_{i+1:03d}.mp4")
        
        return merged_scenes
    
    def save_scene_clips(self, video_path, scenes):
        """Save individual scene video clips
        
        Args:
            video_path: Path to original video
            scenes: List of scene dictionaries
        
        Returns:
            Path to output directory
        """
        video_path = Path(video_path)
        
        # Create output directory
        video_name = video_path.stem
        output_dir = video_path.parent / f"{video_name}_scenes"
        output_dir.mkdir(exist_ok=True)
        
        self._log(f"Saving {len(scenes)} scene clips to: {output_dir}")
        
        # Save each scene
        for i, scene in enumerate(scenes, 1):
            output_path = output_dir / f"scene_{i:03d}.mp4"
            
            # Use ffmpeg to extract scene without re-encoding
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-ss', str(scene['start']),
                '-t', str(scene['duration']),
                '-c', 'copy',  # Copy codecs to avoid re-encoding
                '-avoid_negative_ts', 'make_zero',  # Handle timestamp issues
                '-y', str(output_path)
            ]
            
            self._log(f"  Extracting scene {i}/{len(scenes)}: "
                     f"{scene['start']:.2f}s - {scene['end']:.2f}s")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self._log(f"    Warning: FFmpeg error for scene {i}: {result.stderr}")
                    
                    # Try alternative method with re-encoding if copy fails
                    cmd_alt = [
                        'ffmpeg', '-i', str(video_path),
                        '-ss', str(scene['start']),
                        '-t', str(scene['duration']),
                        '-c:v', 'libx264',
                        '-preset', 'fast',
                        '-crf', '23',
                        '-c:a', 'aac',
                        '-y', str(output_path)
                    ]
                    result = subprocess.run(cmd_alt, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        self._log(f"    ✓ Scene {i} saved (re-encoded)")
                else:
                    # Verify file was created
                    if output_path.exists() and output_path.stat().st_size > 0:
                        self._log(f"    ✓ Scene {i} saved")
                    else:
                        self._log(f"    Warning: Scene {i} file may be empty")
                        
            except Exception as e:
                self._log(f"    Error saving scene {i}: {e}")
        
        # Create scene index file
        index_path = output_dir / "scene_index.txt"
        with open(index_path, 'w') as f:
            f.write(f"Scene Index for {video_name}\n")
            f.write("=" * 50 + "\n\n")
            for i, scene in enumerate(scenes, 1):
                f.write(f"Scene {i:03d}:\n")
                f.write(f"  Start: {scene['start']:.2f}s (frame {scene['frame_start']})\n")
                f.write(f"  End: {scene['end']:.2f}s (frame {scene['frame_end']})\n")
                f.write(f"  Duration: {scene['duration']:.2f}s\n")
                f.write(f"  File: scene_{i:03d}.mp4\n\n")
        
        self._log(f"✓ Scene clips saved to: {output_dir}")
        
        return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Automatically split video into scenes based on visual and audio changes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings (saves clips)
  python scene_splitter.py video.mp4
  
  # High sensitivity (detect more scene changes)
  python scene_splitter.py video.mp4 --sensitivity 0.8
  
  # Set minimum scene duration to 2 seconds
  python scene_splitter.py video.mp4 --min-scene-duration 2.0
  
  # Disable audio analysis for faster processing
  python scene_splitter.py video.mp4 --no-audio
  
  # Only detect scenes without saving clips
  python scene_splitter.py video.mp4 --no-save-clips
  
  # Save output to specific file
  python scene_splitter.py video.mp4 --output scenes.json
  
  # Process every 10th frame for faster analysis
  python scene_splitter.py video.mp4 --sample-rate 10
        """
    )
    
    parser.add_argument('input', help='Input video file')
    parser.add_argument('-o', '--output', help='Output JSON file (default: <input>_scenes.json)')
    parser.add_argument('-s', '--sensitivity', type=float, default=0.5,
                       help='Scene detection sensitivity 0-1 (default: 0.5)')
    parser.add_argument('-m', '--min-scene-duration', type=float, default=1.0,
                       help='Minimum scene duration in seconds (default: 1.0)')
    parser.add_argument('-r', '--sample-rate', type=int, default=5,
                       help='Process every Nth frame (default: 5)')
    parser.add_argument('--no-audio', action='store_true',
                       help='Disable audio analysis')
    parser.add_argument('--no-save-clips', action='store_true',
                       help='Only detect scenes without saving individual clips')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Validate input
    video_path = Path(args.input)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Validate sensitivity
    if not 0 <= args.sensitivity <= 1:
        print(f"❌ Error: Sensitivity must be between 0 and 1")
        sys.exit(1)
    
    # Initialize splitter
    splitter = SceneSplitter(
        sensitivity=args.sensitivity,
        min_scene_duration=args.min_scene_duration,
        sample_rate=args.sample_rate,
        use_audio=not args.no_audio,
        verbose=not args.quiet
    )
    
    # Split scenes
    try:
        scenes = splitter.split_scenes(video_path, save_clips=not args.no_save_clips)
        
        # Generate output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = video_path.with_suffix('').with_suffix('.scenes.json')
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(scenes, f, indent=2)
        
        # Print summary
        print(f"\n✅ Scene detection complete!")
        print(f"   Found {len(scenes)} scenes")
        print(f"   Output: {output_path}")
        
        if not args.no_save_clips and len(scenes) > 0:
            video_name = video_path.stem
            clips_dir = video_path.parent / f"{video_name}_scenes"
            print(f"   Scene clips: {clips_dir}")
        
        if not args.quiet:
            print(f"\nScene list:")
            for i, scene in enumerate(scenes, 1):
                print(f"   {i}. {scene['start']:.2f}s - {scene['end']:.2f}s "
                     f"(duration: {scene['duration']:.2f}s)")
                if 'clip_path' in scene:
                    print(f"      Clip: {Path(scene['clip_path']).name}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during scene detection: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())