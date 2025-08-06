#!/bin/bash

# Setup CloudWatch Events rule to warm up Lambda function
set -e

FUNCTION_NAME="toontune-ai-service"
RULE_NAME="toontune-warmup-rule"
REGION=${AWS_REGION:-"us-east-1"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Setting up Lambda warm-up schedule...${NC}"

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create CloudWatch Events rule (runs every 5 minutes)
echo -e "${YELLOW}Creating CloudWatch Events rule...${NC}"
aws events put-rule \
    --name $RULE_NAME \
    --schedule-expression "rate(5 minutes)" \
    --description "Warm up ToonTune.ai Lambda function" \
    --region $REGION \
    --no-cli-pager

# Add permission for CloudWatch Events to invoke Lambda
echo -e "${YELLOW}Adding permission for CloudWatch Events...${NC}"
aws lambda add-permission \
    --function-name $FUNCTION_NAME \
    --statement-id cloudwatch-warmup \
    --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn "arn:aws:events:$REGION:$AWS_ACCOUNT_ID:rule/$RULE_NAME" \
    --region $REGION \
    --no-cli-pager 2>/dev/null || true

# Create the target (Lambda function)
echo -e "${YELLOW}Creating target for CloudWatch Events rule...${NC}"
cat > warmup-input.json <<EOF
{
  "warmup": true
}
EOF

aws events put-targets \
    --rule $RULE_NAME \
    --targets "Id"="1","Arn"="arn:aws:lambda:$REGION:$AWS_ACCOUNT_ID:function:$FUNCTION_NAME","Input"='{"warmup":true}' \
    --region $REGION \
    --no-cli-pager

# Clean up
rm warmup-input.json

echo -e "${GREEN}Warm-up schedule created successfully!${NC}"
echo -e "${YELLOW}The Lambda function will be warmed up every 5 minutes.${NC}"
echo -e "${YELLOW}To disable warm-up, run:${NC}"
echo -e "aws events delete-rule --name $RULE_NAME --region $REGION"