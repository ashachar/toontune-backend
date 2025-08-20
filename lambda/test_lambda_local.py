#!/usr/bin/env python3
"""
Test Lambda function locally (without AWS dependencies)
"""

import json
import sys
import os
import tempfile
import shutil

# Add to path
sys.path.insert(0, 'python')

# Mock boto3 for local testing
class MockS3Client:
    def upload_file(self, *args, **kwargs):
        print(f"Mock S3 upload: {args[1]}/{args[2]}")
    
    def generate_presigned_url(self, *args, **kwargs):
        return f"https://mock-s3-url.com/{kwargs['Params']['Key']}"
    
    def download_file(self, *args, **kwargs):
        # For testing, just copy a local file
        shutil.copy("../uploads/assets/videos/do_re_mi.mov", args[2])

# Mock boto3
sys.modules['boto3'] = type(sys)('boto3')
sys.modules['boto3'].client = lambda service: MockS3Client()

# Now import the handler
from lambda_handler import lambda_handler

def test_with_local_file():
    """Test with a local video file."""
    
    # Test event with local file
    test_event = {
        "video_url": "../uploads/assets/videos/do_re_mi.mov",
        "text": "LAMBDA"
    }
    
    print("🧪 Testing Lambda function locally")
    print(f"Input: {json.dumps(test_event, indent=2)}")
    print()
    
    # Call handler
    result = lambda_handler(test_event, None)
    
    print("Result:")
    print(json.dumps(result, indent=2))
    
    if result['statusCode'] == 200:
        body = json.loads(result['body'])
        print(f"\n✅ Success!")
        print(f"   Text: {body['text']}")
        print(f"   Output size: {body.get('output_size_mb', 'N/A')} MB")
        print(f"   S3 Key: {body.get('s3_key', 'N/A')}")

if __name__ == "__main__":
    test_with_local_file()