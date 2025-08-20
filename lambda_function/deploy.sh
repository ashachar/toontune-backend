#!/bin/bash

# AWS Lambda Deployment Script for Text Animation Function
# Usage: ./deploy.sh [region] [function-name] [s3-bucket]

set -e

# Configuration
AWS_REGION=${1:-us-east-1}
FUNCTION_NAME=${2:-text-animation-processor}
S3_BUCKET=${3:-text-animation-videos}
ECR_REPO_NAME="text-animation-lambda"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_TAG="latest"

echo "🚀 Deploying Lambda function: ${FUNCTION_NAME}"
echo "   Region: ${AWS_REGION}"
echo "   S3 Bucket: ${S3_BUCKET}"
echo "   ECR Repository: ${ECR_REPO_NAME}"

# Step 1: Create S3 bucket if it doesn't exist
echo "📦 Setting up S3 bucket..."
if ! aws s3 ls "s3://${S3_BUCKET}" 2>/dev/null; then
    echo "Creating S3 bucket: ${S3_BUCKET}"
    if [ "${AWS_REGION}" = "us-east-1" ]; then
        aws s3 mb "s3://${S3_BUCKET}"
    else
        aws s3 mb "s3://${S3_BUCKET}" --region "${AWS_REGION}"
    fi
    
    # Enable public access for processed videos
    aws s3api put-bucket-policy --bucket "${S3_BUCKET}" --policy '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "PublicReadGetObject",
                "Effect": "Allow",
                "Principal": "*",
                "Action": "s3:GetObject",
                "Resource": "arn:aws:s3:::'"${S3_BUCKET}"'/processed/*"
            }
        ]
    }'
else
    echo "S3 bucket already exists: ${S3_BUCKET}"
fi

# Step 2: Create ECR repository if it doesn't exist
echo "🐳 Setting up ECR repository..."
aws ecr describe-repositories --repository-names "${ECR_REPO_NAME}" --region "${AWS_REGION}" 2>/dev/null || \
    aws ecr create-repository --repository-name "${ECR_REPO_NAME}" --region "${AWS_REGION}"

# Step 3: Login to ECR
echo "🔐 Logging in to ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_URI}"

# Step 4: Copy animation utils to lambda_function directory
echo "📋 Copying animation utilities..."
cp -r ../utils ./utils

# Step 5: Build Docker image
echo "🔨 Building Docker image..."
docker build -t "${ECR_REPO_NAME}:${IMAGE_TAG}" .

# Step 6: Tag and push to ECR
echo "📤 Pushing image to ECR..."
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" "${ECR_URI}/${ECR_REPO_NAME}:${IMAGE_TAG}"
docker push "${ECR_URI}/${ECR_REPO_NAME}:${IMAGE_TAG}"

# Step 7: Create IAM role for Lambda if it doesn't exist
echo "👤 Setting up IAM role..."
ROLE_NAME="${FUNCTION_NAME}-role"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

if ! aws iam get-role --role-name "${ROLE_NAME}" 2>/dev/null; then
    echo "Creating IAM role: ${ROLE_NAME}"
    
    # Create role
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
    
    # Attach basic Lambda execution policy
    aws iam attach-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
    
    # Create and attach S3 policy
    aws iam put-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-name "S3Access" \
        --policy-document '{
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
        }'
    
    # Wait for role to be available
    echo "Waiting for IAM role to propagate..."
    sleep 10
else
    echo "IAM role already exists: ${ROLE_NAME}"
fi

# Step 8: Create or update Lambda function
echo "⚡ Deploying Lambda function..."
IMAGE_URI="${ECR_URI}/${ECR_REPO_NAME}:${IMAGE_TAG}"

if aws lambda get-function --function-name "${FUNCTION_NAME}" --region "${AWS_REGION}" 2>/dev/null; then
    echo "Updating existing Lambda function..."
    aws lambda update-function-code \
        --function-name "${FUNCTION_NAME}" \
        --image-uri "${IMAGE_URI}" \
        --region "${AWS_REGION}"
    
    # Wait for update to complete
    aws lambda wait function-updated \
        --function-name "${FUNCTION_NAME}" \
        --region "${AWS_REGION}"
    
    # Update configuration
    aws lambda update-function-configuration \
        --function-name "${FUNCTION_NAME}" \
        --region "${AWS_REGION}" \
        --timeout 300 \
        --memory-size 3008 \
        --environment "Variables={S3_BUCKET=${S3_BUCKET},AWS_REGION=${AWS_REGION}}"
else
    echo "Creating new Lambda function..."
    aws lambda create-function \
        --function-name "${FUNCTION_NAME}" \
        --package-type Image \
        --code ImageUri="${IMAGE_URI}" \
        --role "${ROLE_ARN}" \
        --timeout 300 \
        --memory-size 3008 \
        --environment "Variables={S3_BUCKET=${S3_BUCKET},AWS_REGION=${AWS_REGION}}" \
        --region "${AWS_REGION}"
fi

# Step 9: Create API Gateway (optional)
echo "🌐 Setting up API Gateway..."
API_NAME="${FUNCTION_NAME}-api"
REST_API_ID=$(aws apigateway get-rest-apis --region "${AWS_REGION}" \
    --query "items[?name=='${API_NAME}'].id" --output text)

if [ -z "${REST_API_ID}" ]; then
    echo "Creating API Gateway..."
    REST_API_ID=$(aws apigateway create-rest-api \
        --name "${API_NAME}" \
        --region "${AWS_REGION}" \
        --query 'id' --output text)
    
    # Get root resource
    ROOT_ID=$(aws apigateway get-resources \
        --rest-api-id "${REST_API_ID}" \
        --region "${AWS_REGION}" \
        --query 'items[0].id' --output text)
    
    # Create /process resource
    RESOURCE_ID=$(aws apigateway create-resource \
        --rest-api-id "${REST_API_ID}" \
        --parent-id "${ROOT_ID}" \
        --path-part "process" \
        --region "${AWS_REGION}" \
        --query 'id' --output text)
    
    # Create POST method
    aws apigateway put-method \
        --rest-api-id "${REST_API_ID}" \
        --resource-id "${RESOURCE_ID}" \
        --http-method POST \
        --authorization-type NONE \
        --region "${AWS_REGION}"
    
    # Set up Lambda integration
    aws apigateway put-integration \
        --rest-api-id "${REST_API_ID}" \
        --resource-id "${RESOURCE_ID}" \
        --http-method POST \
        --type AWS_PROXY \
        --integration-http-method POST \
        --uri "arn:aws:apigateway:${AWS_REGION}:lambda:path/2015-03-31/functions/arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}/invocations" \
        --region "${AWS_REGION}"
    
    # Grant API Gateway permission to invoke Lambda
    aws lambda add-permission \
        --function-name "${FUNCTION_NAME}" \
        --statement-id "apigateway-invoke" \
        --action "lambda:InvokeFunction" \
        --principal "apigateway.amazonaws.com" \
        --source-arn "arn:aws:execute-api:${AWS_REGION}:${ACCOUNT_ID}:${REST_API_ID}/*/*" \
        --region "${AWS_REGION}"
    
    # Deploy API
    aws apigateway create-deployment \
        --rest-api-id "${REST_API_ID}" \
        --stage-name "prod" \
        --region "${AWS_REGION}"
    
    API_URL="https://${REST_API_ID}.execute-api.${AWS_REGION}.amazonaws.com/prod/process"
else
    echo "API Gateway already exists"
    API_URL="https://${REST_API_ID}.execute-api.${AWS_REGION}.amazonaws.com/prod/process"
fi

# Cleanup
rm -rf ./utils

echo "✅ Deployment complete!"
echo ""
echo "📋 Summary:"
echo "   Lambda Function: ${FUNCTION_NAME}"
echo "   Lambda ARN: arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}"
echo "   S3 Bucket: ${S3_BUCKET}"
echo "   API Endpoint: ${API_URL}"
echo ""
echo "🧪 Test with:"
echo "curl -X POST ${API_URL} \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"video_url\": \"https://example.com/video.mp4\", \"text\": \"HELLO\"}'"