#!/usr/bin/env python3
"""
Debug script to check what's happening with scaling in Lambda
"""
import json
import boto3

# Initialize clients
lambda_client = boto3.client('lambda', region_name='us-east-1')

# Simple payload to trigger logs
test_payload = {
    "video_url": "s3://toontune-test-videos-1755751835/woman_guitar.mp4",
    "text": "TEST"
}

print("Invoking Lambda for debug...")

# Invoke with log type
response = lambda_client.invoke(
    FunctionName='toontune-text-animation',
    InvocationType='RequestResponse',
    Payload=json.dumps(test_payload),
    LogType='Tail'  # Get logs
)

# Get log result
import base64
if 'LogResult' in response:
    logs = base64.b64decode(response['LogResult']).decode('utf-8')
    print("\n=== LAMBDA LOGS ===")
    # Look for animation phase logs
    for line in logs.split('\n'):
        if 'phase' in line.lower() or 'scale' in line.lower() or 'SHRINK' in line:
            print(line)