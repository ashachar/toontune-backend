#!/bin/bash

# AWS Lambda deployment script for ToonTune.ai services
# This script builds and deploys the Lambda function

set -e

# Configuration
FUNCTION_NAME="toontune-ai-service"
RUNTIME="nodejs18.x"
HANDLER="index.handler"
TIMEOUT=30  # 30 seconds timeout
MEMORY_SIZE=2048  # 2GB memory for image processing
REGION=${AWS_REGION:-"us-east-1"}
ZIP_FILE="lambda_deployment.zip"
ROLE_NAME="toontune-lambda-role"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Lambda deployment process for ToonTune.ai...${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}AWS credentials not configured. Please run 'aws configure'.${NC}"
    exit 1
fi

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${GREEN}AWS Account ID: $AWS_ACCOUNT_ID${NC}"

# Clean up previous build
echo -e "${YELLOW}Cleaning up previous build...${NC}"
rm -rf package/
rm -f $ZIP_FILE

# Create package directory
mkdir -p package

# Copy Lambda function files
echo -e "${YELLOW}Copying Lambda function files...${NC}"
cp index.js package/
cp -r functions package/
cp -r utils package/
# Legacy support
cp lambda_function.js package/ 2>/dev/null || true
cp doodle_generator_lambda.js package/ 2>/dev/null || true

# Copy package.json for dependencies
echo -e "${YELLOW}Creating package.json for Lambda...${NC}"
cat > package/package.json <<EOF
{
  "name": "toontune-lambda",
  "version": "2.0.0",
  "description": "ToonTune.ai Lambda functions",
  "main": "index.js",
  "dependencies": {
    "openai": "^4.20.0",
    "@google/generative-ai": "^0.1.3",
    "replicate": "^0.22.0",
    "sharp": "^0.33.0",
    "aws-sdk": "^2.1400.0",
    "node-fetch": "^2.6.9"
  }
}
EOF

# Install production dependencies
echo -e "${YELLOW}Installing production dependencies...${NC}"
cd package
# Install dependencies except sharp first
npm install --production --omit=dev
# Install sharp specifically for Linux x64
npm install --os=linux --cpu=x64 sharp
cd ..

# Remove unnecessary files
echo -e "${YELLOW}Cleaning up unnecessary files...${NC}"
find package -name "*.md" -delete
find package -name "*.txt" -delete
find package -name ".npmignore" -delete
find package -name ".gitignore" -delete
find package -name "test" -type d -exec rm -rf {} + 2>/dev/null || true
find package -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true

# Create deployment package
echo -e "${YELLOW}Creating deployment package...${NC}"
cd package
zip -r ../$ZIP_FILE . -x "*.git*" "*.DS_Store" "*__pycache__*" "*.pyc"
cd ..

# Get the size of the deployment package
ZIP_SIZE=$(ls -lh $ZIP_FILE | awk '{print $5}')
echo -e "${GREEN}Deployment package created: $ZIP_FILE (Size: $ZIP_SIZE)${NC}"

# Check if IAM role exists, create if not
if aws iam get-role --role-name $ROLE_NAME &> /dev/null; then
    echo -e "${GREEN}IAM role $ROLE_NAME already exists${NC}"
    ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text)
else
    echo -e "${YELLOW}Creating IAM role $ROLE_NAME...${NC}"
    
    # Create trust policy
    cat > trust-policy.json <<EOF
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

    # Create the role
    ROLE_ARN=$(aws iam create-role \
        --role-name $ROLE_NAME \
        --assume-role-policy-document file://trust-policy.json \
        --query 'Role.Arn' \
        --output text)
    
    # Attach basic Lambda execution policy
    aws iam attach-role-policy \
        --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    
    # Clean up trust policy file
    rm trust-policy.json
    
    echo -e "${GREEN}IAM role created: $ROLE_ARN${NC}"
    echo -e "${YELLOW}Waiting for role to propagate...${NC}"
    sleep 10
fi

# Check if Lambda function exists
if aws lambda get-function --function-name $FUNCTION_NAME --region $REGION &> /dev/null; then
    echo -e "${YELLOW}Updating existing Lambda function...${NC}"
    
    # Update function code
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://$ZIP_FILE \
        --region $REGION \
        --no-cli-pager
    
    # Wait for update to complete
    aws lambda wait function-updated \
        --function-name $FUNCTION_NAME \
        --region $REGION
    
    # Update function configuration
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --runtime $RUNTIME \
        --handler $HANDLER \
        --timeout $TIMEOUT \
        --memory-size $MEMORY_SIZE \
        --region $REGION \
        --no-cli-pager
else
    echo -e "${YELLOW}Creating new Lambda function...${NC}"
    
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --runtime $RUNTIME \
        --role $ROLE_ARN \
        --handler $HANDLER \
        --timeout $TIMEOUT \
        --memory-size $MEMORY_SIZE \
        --zip-file fileb://$ZIP_FILE \
        --region $REGION \
        --no-cli-pager
fi

# Update environment variables from .env file
if [ -f "../.env" ]; then
    echo -e "${YELLOW}Setting environment variables from .env file...${NC}"
    
    # Read .env file and format for AWS Lambda
    ENV_VARS="{"
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        if [[ ! "$key" =~ ^# ]] && [[ -n "$key" ]]; then
            # Remove quotes from value if present
            value="${value%\"}"
            value="${value#\"}"
            value="${value%\'}"
            value="${value#\'}"
            
            if [ ${#ENV_VARS} -gt 1 ]; then
                ENV_VARS="${ENV_VARS},"
            fi
            ENV_VARS="${ENV_VARS}\"$key\":\"$value\""
        fi
    done < "../.env"
    ENV_VARS="${ENV_VARS}}"
    
    # Update Lambda environment variables
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --environment "Variables=$ENV_VARS" \
        --region $REGION \
        --no-cli-pager
    
    echo -e "${GREEN}Environment variables updated${NC}"
fi

echo -e "${GREEN}Lambda function deployed successfully!${NC}"

# Create or update API Gateway (optional)
echo -e "${YELLOW}Do you want to create/update an API Gateway for this Lambda? (y/n)${NC}"
read -r CREATE_API

if [[ $CREATE_API == "y" ]]; then
    ./create_api_gateway.sh $FUNCTION_NAME $REGION
fi

# Clean up
echo -e "${YELLOW}Cleaning up...${NC}"
rm -rf package/
# Keep the ZIP file for manual deployment if needed
# rm -f $ZIP_FILE

echo -e "${GREEN}Deployment complete!${NC}"
echo -e "${YELLOW}Lambda function ARN:${NC} arn:aws:lambda:$REGION:$AWS_ACCOUNT_ID:function:$FUNCTION_NAME"
echo -e "${YELLOW}Test your function with:${NC}"
echo -e "aws lambda invoke --function-name $FUNCTION_NAME --payload file://test_payload.json response.json --region $REGION"