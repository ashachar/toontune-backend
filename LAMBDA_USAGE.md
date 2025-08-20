# ToonTune Text Animation Lambda Function

## Overview
This Lambda function applies animated text effects to videos. The animation sequence:
1. **Shrink**: Text starts large and shrinks down
2. **Move Behind**: Text moves behind the main subject/foreground
3. **Dissolve**: Individual letters dissolve away with floating effect

## Lambda Endpoint
```
https://tify4gtckhbsza3o5lxpjhjpc40vetuj.lambda-url.us-east-1.on.aws/
```

## Input/Output Format

### Request Body
```json
{
  "video_url": "s3://bucket/path/to/video.mp4",
  "text": "YOUR TEXT"
}
```

### Response
```json
{
  "video_url": "https://signed-s3-url...",  // Pre-signed URL (7 days)
  "text": "YOUR TEXT",
  "message": "Success",
  "s3_key": "processed/uuid.mp4",          // Permanent S3 location
  "output_size_mb": 1.23
}
```

## Usage Examples

### 1. Simple Python Script
```python
import requests
import json

LAMBDA_URL = "https://tify4gtckhbsza3o5lxpjhjpc40vetuj.lambda-url.us-east-1.on.aws/"

def add_text_animation(video_url, text):
    """Add animated text to a video."""
    
    response = requests.post(
        LAMBDA_URL,
        json={
            "video_url": video_url,
            "text": text
        },
        timeout=120  # 2 minute timeout
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success! Output: {result['video_url']}")
        return result
    else:
        print(f"Error: {response.status_code}")
        return None

# Example usage
result = add_text_animation(
    "s3://toontune-text-animations/test-videos/do_re_mi.mov",
    "HELLO"
)
```

### 2. Command Line with cURL
```bash
# Basic call
curl -X POST https://tify4gtckhbsza3o5lxpjhjpc40vetuj.lambda-url.us-east-1.on.aws/ \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "s3://toontune-text-animations/test-videos/do_re_mi.mov",
    "text": "START"
  }'

# Pretty print with jq
curl -X POST https://tify4gtckhbsza3o5lxpjhjpc40vetuj.lambda-url.us-east-1.on.aws/ \
  -H "Content-Type: application/json" \
  -d '{"video_url": "s3://...", "text": "HELLO"}' | jq .

# Save output URL to file
curl -X POST https://tify4gtckhbsza3o5lxpjhjpc40vetuj.lambda-url.us-east-1.on.aws/ \
  -H "Content-Type: application/json" \
  -d '{"video_url": "s3://...", "text": "HELLO"}' | \
  jq -r '.video_url' > output_url.txt
```

### 3. JavaScript/Node.js
```javascript
const axios = require('axios');

async function addTextAnimation(videoUrl, text) {
    const LAMBDA_URL = 'https://tify4gtckhbsza3o5lxpjhjpc40vetuj.lambda-url.us-east-1.on.aws/';
    
    try {
        const response = await axios.post(LAMBDA_URL, {
            video_url: videoUrl,
            text: text
        });
        
        console.log('Success! Output:', response.data.video_url);
        return response.data;
    } catch (error) {
        console.error('Error:', error.message);
        return null;
    }
}

// Usage
addTextAnimation('s3://bucket/video.mp4', 'HELLO')
    .then(result => console.log(result));
```

### 4. Async/Batch Processing
```python
import asyncio
import aiohttp

async def process_video_async(session, video_url, text):
    """Process a single video asynchronously."""
    url = "https://tify4gtckhbsza3o5lxpjhjpc40vetuj.lambda-url.us-east-1.on.aws/"
    
    async with session.post(url, json={"video_url": video_url, "text": text}) as response:
        if response.status == 200:
            return await response.json()
        return None

async def batch_process_videos(video_text_pairs):
    """Process multiple videos concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_video_async(session, video_url, text)
            for video_url, text in video_text_pairs
        ]
        return await asyncio.gather(*tasks)

# Usage
videos = [
    ("s3://bucket/video1.mp4", "HELLO"),
    ("s3://bucket/video2.mp4", "WORLD"),
    ("s3://bucket/video3.mp4", "ASYNC")
]

results = asyncio.run(batch_process_videos(videos))
```

### 5. Using AWS SDK (boto3)
```python
import boto3
import json

def invoke_with_boto3(video_url, text):
    """Direct Lambda invocation (requires AWS credentials)."""
    
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    response = lambda_client.invoke(
        FunctionName='toontune-text-animation',
        InvocationType='RequestResponse',
        Payload=json.dumps({
            "video_url": video_url,
            "text": text
        })
    )
    
    result = json.loads(response['Payload'].read())
    if isinstance(result.get('body'), str):
        return json.loads(result['body'])
    return result

# Usage (requires AWS credentials configured)
result = invoke_with_boto3(
    "s3://toontune-text-animations/test-videos/do_re_mi.mov",
    "BOTO3"
)
```

### 6. Download Processed Video
```python
import requests

def download_video(video_url, output_path):
    """Download the processed video from S3."""
    
    response = requests.get(video_url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded to: {output_path}")

# After getting result from Lambda
result = add_text_animation("s3://...", "HELLO")
if result:
    download_video(result['video_url'], "output_with_text.mp4")
```

## Complete Working Example
```python
#!/usr/bin/env python3
"""
Complete example: Add text animation to a video and download the result.
"""

import requests
import json
import time

LAMBDA_URL = "https://tify4gtckhbsza3o5lxpjhjpc40vetuj.lambda-url.us-east-1.on.aws/"

def process_video_with_text(video_s3_path, text, output_filename):
    """
    Add animated text to a video and download the result.
    
    Args:
        video_s3_path: S3 path like "s3://bucket/video.mp4"
        text: Text to animate (will be uppercased)
        output_filename: Local filename to save result
    """
    
    print(f"Step 1: Calling Lambda function...")
    print(f"  Video: {video_s3_path}")
    print(f"  Text: {text}")
    
    # Call Lambda
    start_time = time.time()
    response = requests.post(
        LAMBDA_URL,
        json={
            "video_url": video_s3_path,
            "text": text
        },
        timeout=120
    )
    
    if response.status_code != 200:
        print(f"Error: Lambda returned {response.status_code}")
        print(response.text)
        return False
    
    result = response.json()
    elapsed = time.time() - start_time
    
    print(f"Step 2: Lambda processing complete ({elapsed:.1f}s)")
    print(f"  Output size: {result['output_size_mb']} MB")
    print(f"  S3 key: {result['s3_key']}")
    
    # Download the result
    print(f"Step 3: Downloading processed video...")
    video_response = requests.get(result['video_url'], stream=True)
    video_response.raise_for_status()
    
    with open(output_filename, 'wb') as f:
        total_size = int(video_response.headers.get('content-length', 0))
        downloaded = 0
        
        for chunk in video_response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"  Progress: {percent:.1f}%", end='\r')
    
    print(f"\n✓ Success! Video saved to: {output_filename}")
    return True

# Example usage
if __name__ == "__main__":
    success = process_video_with_text(
        video_s3_path="s3://toontune-text-animations/test-videos/do_re_mi.mov",
        text="HELLO WORLD",
        output_filename="video_with_animation.mp4"
    )
    
    if success:
        print("\nVideo processing complete!")
        print("You can now play: video_with_animation.mp4")
```

## Notes

### Supported Video Formats
- Input: MP4, MOV, AVI (most common formats)
- Output: MP4 with H.264 codec (web-compatible)

### Text Animation Details
- Text is automatically converted to uppercase
- Animation duration: ~3.5 seconds total
- Color: Yellow (#FFDC00)
- Position: Centered, slightly above middle

### Performance
- Processing time: 10-25 seconds for most videos
- Max video size: Limited by Lambda timeout (15 min)
- Concurrent requests: Supported

### Error Handling
```python
try:
    response = requests.post(LAMBDA_URL, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()
except requests.Timeout:
    print("Lambda timed out - video might be too large")
except requests.HTTPError as e:
    print(f"HTTP error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### S3 Storage
- Processed videos are stored in: `s3://toontune-text-animations/processed/`
- Pre-signed URLs expire after 7 days
- Original S3 keys (`s3_key` in response) are permanent

## Troubleshooting

### Common Issues

1. **502 Bad Gateway**: Lambda is initializing, retry in a few seconds
2. **Timeout**: Video might be too large, try a shorter clip
3. **Invalid video_url**: Ensure S3 path is correct or URL is accessible
4. **Text not visible**: Check if video has sufficient contrast

### Debug with CloudWatch
```bash
# View Lambda logs
aws logs tail /aws/lambda/toontune-text-animation --follow
```

## Cost Estimation
- Lambda execution: ~$0.0001 per invocation
- S3 storage: ~$0.023 per GB per month
- Data transfer: First 1GB free, then ~$0.09 per GB