#!/bin/bash

# Simple Lambda deployment for text animation
set -e

# Configuration
FUNCTION_NAME="toontune-text-animation"
S3_BUCKET="toontune-text-animations"
AWS_REGION=${AWS_REGION:-us-east-1}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "🚀 Deploying Text Animation Lambda"
echo "   Function: ${FUNCTION_NAME}"
echo "   Bucket: ${S3_BUCKET}"
echo "   Region: ${AWS_REGION}"

# Create deployment package
echo "📦 Creating deployment package..."
rm -rf /tmp/lambda-deploy
mkdir -p /tmp/lambda-deploy

# Copy Python files
cp python/text_animation_processor.py /tmp/lambda-deploy/
cp python/lambda_handler.py /tmp/lambda-deploy/

# Copy utils
cp -r ../utils /tmp/lambda-deploy/

# Create requirements file
cat > /tmp/lambda-deploy/requirements.txt <<EOF
numpy
opencv-python-headless
Pillow
imageio
imageio-ffmpeg
EOF

# Create zip
cd /tmp/lambda-deploy
zip -r9 function.zip . -x "*.pyc" -x "__pycache__/*"
PACKAGE_SIZE=$(du -h function.zip | cut -f1)
echo "Package size: ${PACKAGE_SIZE}"

# Check if IAM role exists, create if not
ROLE_NAME="${FUNCTION_NAME}-role"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

if ! aws iam get-role --role-name "${ROLE_NAME}" 2>/dev/null; then
    echo "👤 Creating IAM role..."
    aws iam create-role \
        --role-name "${ROLE_NAME}" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }'
    
    aws iam attach-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
    
    # S3 permissions
    aws iam put-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-name "S3Access" \
        --policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject"],
                "Resource": "arn:aws:s3:::*/*"
            }]
        }'
    
    sleep 10
fi

# Deploy or update function
if aws lambda get-function --function-name "${FUNCTION_NAME}" 2>/dev/null; then
    echo "⚡ Updating function..."
    aws lambda update-function-code \
        --function-name "${FUNCTION_NAME}" \
        --zip-file fileb://function.zip \
        --region "${AWS_REGION}"
else
    echo "⚡ Creating function..."
    aws lambda create-function \
        --function-name "${FUNCTION_NAME}" \
        --runtime python3.11 \
        --role "${ROLE_ARN}" \
        --handler lambda_handler.lambda_handler \
        --zip-file fileb://function.zip \
        --timeout 300 \
        --memory-size 3008 \
        --environment "Variables={S3_BUCKET=${S3_BUCKET}}" \
        --region "${AWS_REGION}"
fi

# Create Function URL
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

# Add permission
aws lambda add-permission \
    --function-name "${FUNCTION_NAME}" \
    --statement-id FunctionURLAllowPublicAccess \
    --action lambda:InvokeFunctionUrl \
    --principal "*" \
    --function-url-auth-type NONE \
    --region "${AWS_REGION}" 2>/dev/null || true

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Function URL: ${FUNCTION_URL}"
echo ""
echo "Test with:"
echo "curl -X POST ${FUNCTION_URL} \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"video_url\": \"https://example.com/video.mp4\", \"text\": \"HELLO\"}'"