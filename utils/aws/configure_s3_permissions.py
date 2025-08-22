#!/usr/bin/env python3
"""
Configure S3 bucket permissions for public access to specific folders.
This script sets up bucket policy and CORS configuration.
"""

import json
import boto3
import sys
from pathlib import Path
from botocore.exceptions import ClientError

def load_json_config(file_path):
    """Load JSON configuration from file."""
    with open(file_path, 'r') as f:
        return json.dumps(json.load(f))

def configure_bucket_policy(s3_client, bucket_name, policy_file='aws_config/s3_bucket_policy.json'):
    """Configure bucket policy for public read access to specific folders."""
    print(f"Configuring bucket policy for {bucket_name}...")
    
    # Load policy from file
    policy_path = Path(__file__).parent.parent.parent / policy_file
    if not policy_path.exists():
        print(f"Policy file not found: {policy_path}")
        return False
    
    policy_json = load_json_config(policy_path)
    
    # Replace bucket name in policy if needed
    policy_json = policy_json.replace('toontune-text-animations', bucket_name)
    
    try:
        s3_client.put_bucket_policy(
            Bucket=bucket_name,
            Policy=policy_json
        )
        print("✅ Bucket policy configured successfully")
        return True
    except ClientError as e:
        print(f"❌ Error setting bucket policy: {e}")
        return False

def configure_cors(s3_client, bucket_name, cors_file='aws_config/s3_cors_config.json'):
    """Configure CORS for the bucket."""
    print(f"Configuring CORS for {bucket_name}...")
    
    # Load CORS config from file
    cors_path = Path(__file__).parent.parent.parent / cors_file
    if not cors_path.exists():
        print(f"CORS config file not found: {cors_path}")
        return False
    
    with open(cors_path, 'r') as f:
        cors_config = json.load(f)
    
    try:
        s3_client.put_bucket_cors(
            Bucket=bucket_name,
            CORSConfiguration={'CORSRules': cors_config}
        )
        print("✅ CORS configured successfully")
        return True
    except ClientError as e:
        print(f"❌ Error setting CORS: {e}")
        return False

def configure_public_access_block(s3_client, bucket_name):
    """Configure public access block settings to allow bucket policies."""
    print(f"Configuring public access block settings for {bucket_name}...")
    
    try:
        s3_client.put_public_access_block(
            Bucket=bucket_name,
            PublicAccessBlockConfiguration={
                'BlockPublicAcls': True,
                'IgnorePublicAcls': True,
                'BlockPublicPolicy': False,  # Allow bucket policies
                'RestrictPublicBuckets': False  # Allow public bucket policies
            }
        )
        print("✅ Public access block configured")
        return True
    except ClientError as e:
        print(f"❌ Error setting public access block: {e}")
        return False

def test_public_access(bucket_name, test_key='processed/test.txt'):
    """Test public access by uploading a test file and checking access."""
    print(f"\nTesting public access...")
    
    s3_client = boto3.client('s3')
    
    # Upload a test file
    test_content = b"Test file for public access verification"
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=test_key,
            Body=test_content,
            ContentType='text/plain'
        )
        print(f"✅ Test file uploaded: {test_key}")
        
        # Generate public URL
        public_url = f"https://{bucket_name}.s3.amazonaws.com/{test_key}"
        print(f"📎 Public URL: {public_url}")
        
        # Try to access without credentials
        import urllib.request
        try:
            response = urllib.request.urlopen(public_url)
            if response.status == 200:
                print("✅ Public access confirmed!")
                return True
        except urllib.error.HTTPError as e:
            if e.code == 403:
                print("❌ Access denied - permissions not working")
            else:
                print(f"❌ HTTP Error {e.code}: {e.reason}")
            return False
        
    except ClientError as e:
        print(f"❌ Error during test: {e}")
        return False
    finally:
        # Clean up test file
        try:
            s3_client.delete_object(Bucket=bucket_name, Key=test_key)
            print(f"🧹 Test file cleaned up")
        except:
            pass

def main():
    """Main function to configure S3 bucket permissions."""
    
    # Default bucket name
    bucket_name = 'toontune-text-animations'
    
    # Allow override from command line
    if len(sys.argv) > 1:
        bucket_name = sys.argv[1]
    
    print(f"🪣 Configuring S3 bucket: {bucket_name}")
    print("=" * 50)
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    # Step 1: Configure public access block
    if not configure_public_access_block(s3_client, bucket_name):
        print("Failed to configure public access block. Continuing...")
    
    # Step 2: Configure bucket policy
    if not configure_bucket_policy(s3_client, bucket_name):
        print("Failed to configure bucket policy")
        return 1
    
    # Step 3: Configure CORS
    if not configure_cors(s3_client, bucket_name):
        print("Failed to configure CORS")
        return 1
    
    # Step 4: Test public access
    print("\n" + "=" * 50)
    if test_public_access(bucket_name):
        print("\n✨ All configurations applied successfully!")
        print(f"Your videos in /processed/* and /assets/* are now publicly accessible")
        print(f"Example URL format: https://{bucket_name}.s3.amazonaws.com/processed/your-video.mp4")
        return 0
    else:
        print("\n⚠️ Configuration applied but public access test failed")
        print("Please check AWS Console for any additional settings needed")
        return 1

if __name__ == "__main__":
    sys.exit(main())