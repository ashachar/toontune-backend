#!/bin/bash

# Create API Gateway for Lambda function
set -e

FUNCTION_NAME=${1:-"toontune-ai-service"}
REGION=${2:-"us-east-1"}
API_NAME="toontune-api"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Creating API Gateway for $FUNCTION_NAME...${NC}"

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Check if API Gateway already exists
API_ID=$(aws apigatewayv2 get-apis --region $REGION --query "Items[?Name=='$API_NAME'].ApiId" --output text 2>/dev/null || echo "")

if [ -z "$API_ID" ]; then
    echo -e "${YELLOW}Creating new HTTP API...${NC}"
    
    # Create HTTP API
    API_ID=$(aws apigatewayv2 create-api \
        --name $API_NAME \
        --protocol-type HTTP \
        --cors-configuration '{"AllowOrigins":["*"],"AllowMethods":["*"],"AllowHeaders":["*"]}' \
        --region $REGION \
        --query ApiId \
        --output text)
    
    echo -e "${GREEN}API created with ID: $API_ID${NC}"
else
    echo -e "${GREEN}Using existing API with ID: $API_ID${NC}"
fi

# Create Lambda integration
echo -e "${YELLOW}Creating Lambda integration...${NC}"
INTEGRATION_ID=$(aws apigatewayv2 create-integration \
    --api-id $API_ID \
    --integration-type AWS_PROXY \
    --integration-uri "arn:aws:lambda:$REGION:$AWS_ACCOUNT_ID:function:$FUNCTION_NAME" \
    --payload-format-version 2.0 \
    --region $REGION \
    --query IntegrationId \
    --output text)

echo -e "${GREEN}Integration created with ID: $INTEGRATION_ID${NC}"

# Create route
echo -e "${YELLOW}Creating route...${NC}"
ROUTE_ID=$(aws apigatewayv2 create-route \
    --api-id $API_ID \
    --route-key 'POST /ai' \
    --target "integrations/$INTEGRATION_ID" \
    --region $REGION \
    --query RouteId \
    --output text)

echo -e "${GREEN}Route created with ID: $ROUTE_ID${NC}"

# Create deployment
echo -e "${YELLOW}Creating deployment...${NC}"
STAGE_NAME="prod"
aws apigatewayv2 create-stage \
    --api-id $API_ID \
    --stage-name $STAGE_NAME \
    --auto-deploy \
    --region $REGION \
    --no-cli-pager 2>/dev/null || \
aws apigatewayv2 update-stage \
    --api-id $API_ID \
    --stage-name $STAGE_NAME \
    --auto-deploy \
    --region $REGION \
    --no-cli-pager

# Grant API Gateway permission to invoke Lambda
echo -e "${YELLOW}Granting API Gateway permission to invoke Lambda...${NC}"
aws lambda add-permission \
    --function-name $FUNCTION_NAME \
    --statement-id apigateway-invoke \
    --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com \
    --source-arn "arn:aws:execute-api:$REGION:$AWS_ACCOUNT_ID:$API_ID/*/*" \
    --region $REGION \
    --no-cli-pager 2>/dev/null || true

# Get the API endpoint
API_ENDPOINT=$(aws apigatewayv2 get-api \
    --api-id $API_ID \
    --region $REGION \
    --query ApiEndpoint \
    --output text)

echo -e "${GREEN}API Gateway setup complete!${NC}"
echo -e "${YELLOW}API Endpoint:${NC} $API_ENDPOINT/$STAGE_NAME/ai"
echo -e "${YELLOW}Test with:${NC}"
echo -e "curl -X POST $API_ENDPOINT/$STAGE_NAME/ai \\"
echo -e "  -H 'Content-Type: application/json' \\"
echo -e "  -d '{\"action\":\"generate-doodle\",\"description\":\"cute cat\",\"provider\":\"mock\"}'"