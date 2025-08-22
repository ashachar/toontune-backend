#!/usr/bin/env python3
"""
Test Lambda function with the woman playing guitar video
"""
import json
import boto3
import requests
import time

# Initialize clients
lambda_client = boto3.client('lambda', region_name='us-east-1')
s3_client = boto3.client('s3', region_name='us-east-1')

# Generate fresh presigned URL for woman guitar video
bucket = 'toontune-test-videos-1755751835'
key = 'woman_guitar.mp4'
video_url = s3_client.generate_presigned_url(
    'get_object',
    Params={'Bucket': bucket, 'Key': key},
    ExpiresIn=3600
)

# Create test payload
test_payload = {
    "video_url": video_url,
    "text": "HELLO WORLD",
    "position": [640, 320],  # Center position for 1280x720
    "start_frame": 30,
    "duration": 4.0
}

print("=" * 60)
print("Testing Lambda with Woman Playing Guitar Video")
print("=" * 60)
print(f"\nInput: woman_guitar.mp4 (3 seconds)")
print(f"Text: {test_payload['text']}")
print(f"Position: {test_payload['position']}")
print(f"\nInvoking Lambda function...")

start_time = time.time()

# Invoke Lambda function
response = lambda_client.invoke(
    FunctionName='toontune-text-animation',
    InvocationType='RequestResponse',
    Payload=json.dumps(test_payload)
)

# Parse response
status_code = response['StatusCode']
payload = json.loads(response['Payload'].read())

elapsed = time.time() - start_time
print(f"Processing time: {elapsed:.1f} seconds")

if status_code == 200 and 'body' in payload:
    body = json.loads(payload['body'])
    if 'video_url' in body:
        print(f"\n✅ SUCCESS!")
        print(f"Output size: {body.get('output_size_mb', 'Unknown')} MB")
        print(f"S3 key: {body.get('s3_key', 'Unknown')}")
        
        # Download the video
        print("\nDownloading output video...")
        r = requests.get(body['video_url'])
        output_filename = 'woman_guitar_with_text.mp4'
        with open(output_filename, 'wb') as f:
            f.write(r.content)
        print(f"✅ Video saved as: {output_filename}")
        print(f"\nTo view: open {output_filename}")
    else:
        print(f"\n❌ Failed: {body.get('error', 'Unknown error')}")
        if 'error' in body:
            print(f"Error details: {body['error']}")
else:
    print(f"\n❌ Failed with status code: {status_code}")
    if 'errorMessage' in payload:
        print(f"Error: {payload['errorMessage']}")