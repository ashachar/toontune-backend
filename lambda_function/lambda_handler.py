#!/usr/bin/env python3
"""
AWS Lambda handler for text animation processing.
Accepts a video URL and text, returns animated video with text effect.
"""

import json
import os
import sys
import boto3
import tempfile
import urllib.request
from pathlib import Path
import traceback
import numpy as np
import cv2
from typing import Dict, Any
import uuid

# Add parent directory to path for imports
sys.path.insert(0, '/opt/python')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import animation modules
from utils.animations.text_behind_segment import TextBehindSegment
from utils.animations.word_dissolve import WordDissolve

# Initialize AWS clients
s3_client = boto3.client('s3')

# Configuration from environment
S3_BUCKET = os.environ.get('S3_BUCKET', 'text-animation-videos')
S3_REGION = os.environ.get('AWS_REGION', 'us-east-1')
TEMP_DIR = '/tmp'


def download_video(video_url: str) -> str:
    """Download video from URL to temp file."""
    temp_path = os.path.join(TEMP_DIR, f"input_{uuid.uuid4().hex}.mp4")
    
    if video_url.startswith('s3://'):
        # Parse S3 URL
        parts = video_url.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        s3_client.download_file(bucket, key, temp_path)
    else:
        # Download from HTTP/HTTPS
        urllib.request.urlretrieve(video_url, temp_path)
    
    return temp_path


def upload_to_s3(local_path: str, key: str) -> str:
    """Upload file to S3 and return public URL."""
    s3_client.upload_file(
        local_path, 
        S3_BUCKET, 
        key,
        ExtraArgs={'ContentType': 'video/mp4'}
    )
    
    # Generate public URL
    url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"
    return url


def process_video(input_path: str, text: str, output_path: str) -> None:
    """
    Process video with text animation combo.
    1. TextBehindSegment: Shrink and move behind
    2. WordDissolve: Dissolve letters
    """
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Animation parameters
    center_position = (width // 2, int(height * 0.45))
    font_size = int(min(150, height * 0.28))
    
    # Phase durations
    phase1_frames = 30  # Shrinking
    phase2_frames = 20  # Moving behind
    phase3_frames = 40  # Stable behind
    dissolve_frames = 60  # Dissolve
    
    total_animation_frames = phase1_frames + phase2_frames + phase3_frames + dissolve_frames
    
    # Create TextBehindSegment animation
    text_animator = TextBehindSegment(
        element_path=input_path,
        background_path=input_path,
        position=center_position,
        text=text.upper(),
        font_size=font_size,
        text_color=(255, 220, 0),  # Yellow
        start_scale=2.0,
        end_scale=1.0,
        phase1_duration=phase1_frames / fps,
        phase2_duration=phase2_frames / fps,
        phase3_duration=phase3_frames / fps,
        center_position=center_position,
        fps=fps
    )
    
    # Get handoff data after stable phase
    handoff_frame = phase1_frames + phase2_frames + phase3_frames - 1
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    cap.release()
    
    if frame is not None:
        # Render the handoff frame to freeze the text state
        _ = text_animator.render_text_frame(frame, handoff_frame)
    
    handoff_data = text_animator.get_handoff_data()
    
    # Create WordDissolve animation with handoff
    word_dissolver = WordDissolve(
        element_path=input_path,
        background_path=input_path,
        position=center_position,
        word=text.upper(),
        font_size=font_size,
        text_color=(255, 220, 0),
        stable_duration=0.17,  # 5 frames at 30fps
        dissolve_duration=2.0,  # 60 frames at 30fps
        dissolve_stagger=0.5,
        float_distance=30,
        randomize_order=False,
        maintain_kerning=True,
        center_position=center_position,
        handoff_data=handoff_data,
        fps=fps
    )
    
    # Process video frames
    cap = cv2.VideoCapture(input_path)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply animations based on frame index
        if frame_idx < phase1_frames + phase2_frames + phase3_frames:
            # TextBehindSegment phase
            frame = text_animator.render_text_frame(frame, frame_idx)
        elif frame_idx < total_animation_frames:
            # WordDissolve phase
            dissolve_frame = frame_idx - (phase1_frames + phase2_frames + phase3_frames)
            frame = word_dissolver.render_word_frame(frame, dissolve_frame)
        
        out.write(frame)
        frame_idx += 1
        
        # Stop after animation completes
        if frame_idx >= total_animation_frames:
            break
    
    # Write remaining frames without animation if needed
    while frame_idx < total_frames and frame_idx < total_animation_frames + 30:  # Add 1 second buffer
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function.
    
    Expected event format:
    {
        "video_url": "https://example.com/video.mp4" or "s3://bucket/key.mp4",
        "text": "START"
    }
    
    Returns:
    {
        "statusCode": 200,
        "body": {
            "video_url": "https://output-bucket.s3.region.amazonaws.com/processed/uuid.mp4",
            "message": "Success"
        }
    }
    """
    
    try:
        # Parse input
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event
        
        video_url = body.get('video_url')
        text = body.get('text', 'START')
        
        if not video_url:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'video_url is required'})
            }
        
        # Generate unique output key
        output_key = f"processed/{uuid.uuid4().hex}.mp4"
        
        # Download input video
        print(f"Downloading video from {video_url}")
        input_path = download_video(video_url)
        
        # Process video
        output_path = os.path.join(TEMP_DIR, f"output_{uuid.uuid4().hex}.mp4")
        print(f"Processing video with text: {text}")
        process_video(input_path, text, output_path)
        
        # Upload result
        print(f"Uploading to S3: {output_key}")
        output_url = upload_to_s3(output_path, output_key)
        
        # Cleanup temp files
        os.remove(input_path)
        os.remove(output_path)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'video_url': output_url,
                'message': 'Success',
                'text': text
            })
        }
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        print(traceback.format_exc())
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Failed to process video'
            })
        }


# For local testing
if __name__ == "__main__":
    test_event = {
        "video_url": "https://example.com/test.mp4",
        "text": "HELLO"
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))