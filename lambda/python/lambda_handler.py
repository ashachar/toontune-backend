#!/usr/bin/env python3
"""
Lambda handler for text animation processing
"""

import json
import os
import sys
import boto3
import tempfile
import urllib.request
from pathlib import Path
import traceback
import uuid

# Import our processor
from text_animation_processor import process_video

s3_client = boto3.client('s3')
S3_BUCKET = os.environ.get('S3_BUCKET', 'toontune-text-animations')
S3_REGION = os.environ.get('AWS_REGION', 'us-east-1')

def download_video(video_url):
    """Download video from URL or local path."""
    temp_path = f"/tmp/input_{uuid.uuid4().hex}.mp4"
    
    if video_url.startswith('s3://'):
        parts = video_url.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        s3_client.download_file(bucket, key, temp_path)
    elif os.path.exists(video_url):
        # Local file for testing
        import shutil
        shutil.copy(video_url, temp_path)
    else:
        urllib.request.urlretrieve(video_url, temp_path)
    
    return temp_path

def upload_to_s3(local_path, key):
    """Upload to S3 and return pre-signed URL."""
    s3_client.upload_file(
        local_path, 
        S3_BUCKET, 
        key,
        ExtraArgs={
            'ContentType': 'video/mp4',
            'CacheControl': 'max-age=86400',  # Cache for 1 day
            'ContentDisposition': 'inline'  # Display inline in browser
        }
    )
    # Generate pre-signed URL valid for 7 days
    url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': S3_BUCKET, 'Key': key},
        ExpiresIn=604800  # 7 days
    )
    return url

def lambda_handler(event, context):
    """Lambda handler."""
    try:
        # Parse input
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event
        
        video_url = body.get('video_url')
        text = body.get('text', 'START').upper()
        
        if not video_url:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'video_url is required'})
            }
        
        # Process
        print(f"Processing: {video_url} with text: {text}")
        input_path = download_video(video_url)
        output_path = f"/tmp/output_{uuid.uuid4().hex}.mp4"
        
        process_video(input_path, text, output_path)
        
        # Upload
        output_key = f"processed/{uuid.uuid4().hex}.mp4"
        output_url = upload_to_s3(output_path, output_key)
        
        # Get file size
        output_size = os.path.getsize(output_path) / (1024 * 1024)
        
        # Cleanup
        os.remove(input_path)
        os.remove(output_path)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'video_url': output_url,
                'text': text,
                'message': 'Success',
                's3_key': output_key,
                'output_size_mb': round(output_size, 2)
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }