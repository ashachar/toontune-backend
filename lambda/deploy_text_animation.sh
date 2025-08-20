#!/bin/bash

# Deploy Text Animation Lambda Function as a Lambda Layer with Python dependencies
# This script creates a containerized Lambda function with text animation capabilities

set -e

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
FUNCTION_NAME="toontune-text-animation"
LAYER_NAME="text-animation-python-deps"
S3_BUCKET="toontune-text-animations"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "🚀 Deploying Text Animation Lambda Function"
echo "   Region: ${AWS_REGION}"
echo "   Function: ${FUNCTION_NAME}"
echo "   Account: ${ACCOUNT_ID}"
echo ""

# Step 1: Create S3 bucket if it doesn't exist
echo "📦 Setting up S3 bucket..."
if ! aws s3 ls "s3://${S3_BUCKET}" 2>/dev/null; then
    echo "Creating S3 bucket: ${S3_BUCKET}"
    if [ "${AWS_REGION}" = "us-east-1" ]; then
        aws s3 mb "s3://${S3_BUCKET}"
    else
        aws s3 mb "s3://${S3_BUCKET}" --region "${AWS_REGION}"
    fi
    
    # Note: Public access blocked by default - use pre-signed URLs instead
    echo "Note: Bucket created with private access. Using pre-signed URLs for output."
else
    echo "S3 bucket already exists: ${S3_BUCKET}"
fi

# Step 2: Create Python dependencies layer
echo "📚 Building Python dependencies layer..."
mkdir -p /tmp/lambda-layer/python

# Create requirements file
cat > /tmp/lambda-layer/requirements.txt <<EOF
numpy==1.24.3
opencv-python-headless==4.8.1.78
Pillow==10.1.0
imageio==2.31.1
imageio-ffmpeg==0.4.9
EOF

# Install dependencies
pip3 install -r /tmp/lambda-layer/requirements.txt -t /tmp/lambda-layer/python --platform manylinux2014_x86_64 --only-binary=:all: --python-version 3.11

# Copy our animation modules to the layer
cp -r ../utils /tmp/lambda-layer/python/

# Create layer zip
cd /tmp/lambda-layer
zip -r9 text-animation-layer.zip python
LAYER_SIZE=$(du -h text-animation-layer.zip | cut -f1)
echo "Layer size: ${LAYER_SIZE}"

# Upload layer to S3
aws s3 cp text-animation-layer.zip "s3://${S3_BUCKET}/layers/text-animation-layer.zip"

# Publish Lambda layer
echo "🎯 Publishing Lambda layer..."
LAYER_VERSION=$(aws lambda publish-layer-version \
    --layer-name "${LAYER_NAME}" \
    --description "Python dependencies for text animation" \
    --content S3Bucket="${S3_BUCKET}",S3Key="layers/text-animation-layer.zip" \
    --compatible-runtimes python3.11 \
    --region "${AWS_REGION}" \
    --query 'Version' \
    --output text)

LAYER_ARN="arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:layer:${LAYER_NAME}:${LAYER_VERSION}"
echo "Layer ARN: ${LAYER_ARN}"

# Step 3: Create IAM role if it doesn't exist
echo "👤 Setting up IAM role..."
ROLE_NAME="${FUNCTION_NAME}-role"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

if ! aws iam get-role --role-name "${ROLE_NAME}" 2>/dev/null; then
    echo "Creating IAM role: ${ROLE_NAME}"
    
    # Create trust policy
    cat > /tmp/trust-policy.json <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
    
    aws iam create-role \
        --role-name "${ROLE_NAME}" \
        --assume-role-policy-document file:///tmp/trust-policy.json
    
    # Attach basic execution policy
    aws iam attach-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
    
    # Create S3 access policy
    cat > /tmp/s3-policy.json <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:PutObjectAcl"
            ],
            "Resource": "arn:aws:s3:::*/*"
        },
        {
            "Effect": "Allow",
            "Action": "s3:ListBucket",
            "Resource": "arn:aws:s3:::*"
        }
    ]
}
EOF
    
    aws iam put-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-name "S3Access" \
        --policy-document file:///tmp/s3-policy.json
    
    rm /tmp/trust-policy.json /tmp/s3-policy.json
    
    echo "Waiting for IAM role to propagate..."
    sleep 10
else
    echo "IAM role already exists: ${ROLE_NAME}"
fi

# Step 4: Create deployment package
echo "📦 Creating deployment package..."
mkdir -p /tmp/lambda-deploy
cp python/text_animation_processor.py /tmp/lambda-deploy/
cp -r ../utils /tmp/lambda-deploy/

# Create Lambda handler wrapper
cat > /tmp/lambda-deploy/lambda_function.py <<'EOF'
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
    """Download video from URL."""
    temp_path = f"/tmp/input_{uuid.uuid4().hex}.mp4"
    
    if video_url.startswith('s3://'):
        parts = video_url.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        s3_client.download_file(bucket, key, temp_path)
    else:
        urllib.request.urlretrieve(video_url, temp_path)
    
    return temp_path

def upload_to_s3(local_path, key):
    """Upload to S3 and return pre-signed URL."""
    s3_client.upload_file(
        local_path, 
        S3_BUCKET, 
        key,
        ExtraArgs={'ContentType': 'video/mp4'}
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
        
        # Cleanup
        os.remove(input_path)
        os.remove(output_path)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'video_url': output_url,
                'text': text,
                'message': 'Success'
            })
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
EOF

# Create zip
cd /tmp/lambda-deploy
zip -r9 lambda-function.zip .
DEPLOY_SIZE=$(du -h lambda-function.zip | cut -f1)
echo "Deployment package size: ${DEPLOY_SIZE}"

# Step 5: Create or update Lambda function
echo "⚡ Deploying Lambda function..."
if aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}" 2>/dev/null; then
    echo "Updating existing function..."
    aws lambda update-function-code \
        --function-name "${FUNCTION_NAME}" \
        --zip-file fileb://lambda-function.zip \
        --region "${AWS_REGION}"
    
    aws lambda wait function-updated \
        --function-name "${FUNCTION_NAME}" \
        --region "${AWS_REGION}"
    
    # Update configuration
    aws lambda update-function-configuration \
        --function-name "${FUNCTION_NAME}" \
        --runtime python3.11 \
        --handler lambda_function.lambda_handler \
        --timeout 300 \
        --memory-size 3008 \
        --layers "${LAYER_ARN}" \
        --environment "Variables={S3_BUCKET=${S3_BUCKET},AWS_REGION=${AWS_REGION}}" \
        --region "${AWS_REGION}"
else
    echo "Creating new function..."
    aws lambda create-function \
        --function-name "${FUNCTION_NAME}" \
        --runtime python3.11 \
        --role "${ROLE_ARN}" \
        --handler lambda_function.lambda_handler \
        --zip-file fileb://lambda-function.zip \
        --timeout 300 \
        --memory-size 3008 \
        --layers "${LAYER_ARN}" \
        --environment "Variables={S3_BUCKET=${S3_BUCKET},AWS_REGION=${AWS_REGION}}" \
        --region "${AWS_REGION}"
fi

# Step 6: Create Function URL for easy testing
echo "🌐 Setting up Function URL..."
FUNCTION_URL=$(aws lambda create-function-url-config \
    --function-name "${FUNCTION_NAME}" \
    --auth-type NONE \
    --region "${AWS_REGION}" \
    --query 'FunctionUrl' \
    --output text 2>/dev/null || \
    aws lambda get-function-url-config \
    --function-name "${FUNCTION_NAME}" \
    --region "${AWS_REGION}" \
    --query 'FunctionUrl' \
    --output text)

# Add permission for Function URL
aws lambda add-permission \
    --function-name "${FUNCTION_NAME}" \
    --statement-id FunctionURLAllowPublicAccess \
    --action lambda:InvokeFunctionUrl \
    --principal "*" \
    --function-url-auth-type NONE \
    --region "${AWS_REGION}" 2>/dev/null || true

# Cleanup
cd /
rm -rf /tmp/lambda-layer /tmp/lambda-deploy

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📋 Summary:"
echo "   Function Name: ${FUNCTION_NAME}"
echo "   Function ARN: arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}"
echo "   S3 Bucket: ${S3_BUCKET}"
echo "   Function URL: ${FUNCTION_URL}"
echo ""
echo "🧪 Test with curl:"
echo "curl -X POST ${FUNCTION_URL} \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"video_url\": \"https://example.com/video.mp4\", \"text\": \"HELLO\"}'"
echo ""
echo "🧪 Test with AWS CLI:"
echo "aws lambda invoke \\"
echo "  --function-name ${FUNCTION_NAME} \\"
echo "  --payload '{\"video_url\":\"s3://bucket/video.mp4\",\"text\":\"HELLO\"}' \\"
echo "  --region ${AWS_REGION} \\"
echo "  response.json"