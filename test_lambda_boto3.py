#!/usr/bin/env python3
"""
Example using boto3 to invoke Lambda function directly (requires AWS credentials).
"""

import boto3
import json

def invoke_lambda_with_boto3(video_url, text):
    """
    Invoke Lambda using boto3 (requires AWS credentials configured).
    
    This method is useful if you want to invoke the Lambda directly
    without going through the Function URL.
    """
    
    # Create Lambda client
    lambda_client = boto3.client('lambda', region_name='us-east-1')
    
    # Prepare payload
    payload = {
        "video_url": video_url,
        "text": text
    }
    
    try:
        # Invoke the Lambda function
        response = lambda_client.invoke(
            FunctionName='toontune-text-animation',
            InvocationType='RequestResponse',  # Synchronous execution
            Payload=json.dumps(payload)
        )
        
        # Parse the response
        result = json.loads(response['Payload'].read())
        
        if response['StatusCode'] == 200:
            # Parse the body if it's a string
            if isinstance(result.get('body'), str):
                body = json.loads(result['body'])
            else:
                body = result
            
            print(f"✓ Success!")
            print(f"  Video URL: {body['video_url']}")
            print(f"  S3 Key: {body['s3_key']}")
            return body
        else:
            print(f"✗ Lambda returned status code: {response['StatusCode']}")
            return None
            
    except Exception as e:
        print(f"✗ Error invoking Lambda: {e}")
        return None


def main():
    """Example usage with boto3."""
    
    print("Using boto3 to invoke Lambda directly...")
    print("Note: This requires AWS credentials configured (aws configure)")
    print("-" * 60)
    
    result = invoke_lambda_with_boto3(
        video_url="s3://toontune-text-animations/test-videos/do_re_mi.mov",
        text="BOTO3"
    )
    
    if result:
        print(f"\nProcessed video available at:")
        print(f"{result['video_url']}")


if __name__ == "__main__":
    main()