import json
import boto3
import base64
import time

# Initialize clients
lambda_client = boto3.client('lambda', region_name='us-east-1')
s3_client = boto3.client('s3', region_name='us-east-1')

# Generate fresh presigned URL for test video
bucket = 'toontune-test-videos-1755751835'
key = 'test_input.mp4'
video_url = s3_client.generate_presigned_url(
    'get_object',
    Params={'Bucket': bucket, 'Key': key},
    ExpiresIn=3600
)

# Create test payload
test_payload = {
    "video_url": video_url,
    "text": "HELLO WORLD",
    "position": [600, 300],
    "start_frame": 30,
    "duration": 4.0
}

print("Testing Lambda function: toontune-text-animation")
print(f"Payload: {json.dumps(test_payload, indent=2)}")

# Invoke Lambda function
response = lambda_client.invoke(
    FunctionName='toontune-text-animation',
    InvocationType='RequestResponse',
    Payload=json.dumps(test_payload)
)

# Parse response
status_code = response['StatusCode']
payload = json.loads(response['Payload'].read())

print(f"\nResponse Status: {status_code}")
print(f"Response: {json.dumps(payload, indent=2)}")

if status_code == 200:
    if 'body' in payload:
        body = json.loads(payload['body'])
        if 'video_url' in body:
            print(f"\n✅ Success! Output video: {body['video_url']}")
            print(f"Output size: {body.get('output_size_mb', 'Unknown')} MB")
            print(f"S3 key: {body.get('s3_key', 'Unknown')}")
            # Download the video
            import requests
            r = requests.get(body['video_url'])
            with open('lambda_test_output.mp4', 'wb') as f:
                f.write(r.content)
            print("\nVideo downloaded to: lambda_test_output.mp4")
        else:
            print(f"\n❌ Failed: {body.get('error', 'Unknown error')}")
    else:
        print(f"\n❌ Failed: {payload.get('errorMessage', 'Unknown error')}")
else:
    print(f"\n❌ Failed with status code: {status_code}")