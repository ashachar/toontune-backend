# Text Animation Lambda Function

AWS Lambda function that applies text animation effects to videos. The animation includes:
1. Text shrinking from 2x to 1x size
2. Text moving behind the subject
3. Letters dissolving with floating effect

## Features

- Accepts video URL (HTTP/HTTPS or S3)
- Processes text animation combo (shrink → behind → dissolve)
- Returns URL to processed video on S3
- Containerized deployment using Docker
- API Gateway integration for REST endpoint

## Prerequisites

- AWS CLI configured with credentials
- Docker installed
- Python 3.11+
- AWS account with permissions for:
  - Lambda
  - ECR (Elastic Container Registry)
  - S3
  - IAM
  - API Gateway

## Project Structure

```
lambda_function/
├── lambda_handler.py      # Main Lambda function code
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container definition
├── deploy.sh              # Deployment script
├── test_local.py          # Local testing script
└── utils/                 # Animation modules (copied during build)
```

## Deployment

### Quick Deploy

```bash
# Deploy with defaults (us-east-1, auto-generated names)
./deploy.sh

# Deploy with custom settings
./deploy.sh us-west-2 my-animation-function my-video-bucket
```

### Manual Deployment Steps

1. **Configure AWS region and names:**
```bash
export AWS_REGION=us-east-1
export FUNCTION_NAME=text-animation-processor
export S3_BUCKET=text-animation-videos
```

2. **Run deployment:**
```bash
cd lambda_function
./deploy.sh $AWS_REGION $FUNCTION_NAME $S3_BUCKET
```

3. **The script will:**
   - Create S3 bucket for video storage
   - Create ECR repository for Docker image
   - Build and push Docker container
   - Create IAM role with necessary permissions
   - Deploy Lambda function
   - Set up API Gateway endpoint

## Usage

### Via API Gateway (REST)

```bash
curl -X POST https://your-api-id.execute-api.us-east-1.amazonaws.com/prod/process \
  -H 'Content-Type: application/json' \
  -d '{
    "video_url": "https://example.com/input.mp4",
    "text": "HELLO"
  }'
```

### Via AWS SDK (Python)

```python
import boto3
import json

lambda_client = boto3.client('lambda', region_name='us-east-1')

payload = {
    "video_url": "https://example.com/input.mp4",
    "text": "HELLO"
}

response = lambda_client.invoke(
    FunctionName='text-animation-processor',
    InvocationType='RequestResponse',
    Payload=json.dumps(payload)
)

result = json.loads(response['Payload'].read())
print(f"Processed video: {result['body']['video_url']}")
```

### Via AWS CLI

```bash
aws lambda invoke \
  --function-name text-animation-processor \
  --payload '{"video_url":"https://example.com/input.mp4","text":"HELLO"}' \
  response.json
```

## Input Format

```json
{
  "video_url": "https://example.com/video.mp4",  // or "s3://bucket/key.mp4"
  "text": "HELLO"                                 // Text to animate (will be uppercased)
}
```

## Output Format

```json
{
  "statusCode": 200,
  "body": {
    "video_url": "https://bucket.s3.region.amazonaws.com/processed/uuid.mp4",
    "message": "Success",
    "text": "HELLO"
  }
}
```

## Configuration

### Lambda Settings
- **Timeout:** 5 minutes (300 seconds)
- **Memory:** 3008 MB
- **Container Image:** Python 3.11 with OpenCV

### Environment Variables
- `S3_BUCKET`: Output bucket for processed videos
- `AWS_REGION`: AWS region for S3 operations

## Local Testing

```bash
# Test the processing logic locally
python test_local.py

# This will:
# 1. Use a local test video
# 2. Process with text "HELLO"
# 3. Create output.mp4 and output.gif
```

## Monitoring

View Lambda logs:
```bash
aws logs tail /aws/lambda/text-animation-processor --follow
```

Check Lambda metrics:
```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=text-animation-processor \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-12-31T23:59:59Z \
  --period 3600 \
  --statistics Average
```

## Cost Estimation

- **Lambda:** ~$0.0000166667 per GB-second
- **S3 Storage:** ~$0.023 per GB/month
- **S3 Requests:** ~$0.0004 per 1000 GET requests
- **API Gateway:** ~$3.50 per million requests

Example: Processing 1000 videos/month (5 seconds each, 50MB average):
- Lambda: ~$0.25
- S3 Storage: ~$1.15
- S3 Requests: ~$0.01
- **Total:** ~$1.41/month

## Troubleshooting

### Common Issues

1. **Timeout errors:**
   - Increase Lambda timeout (max 15 minutes)
   - Optimize video processing or use smaller videos

2. **Memory errors:**
   - Increase Lambda memory (max 10,240 MB)
   - Process videos in chunks

3. **S3 access denied:**
   - Check IAM role permissions
   - Ensure bucket policy allows Lambda access

4. **Docker build fails:**
   - Ensure Docker daemon is running
   - Check available disk space
   - Verify ECR login credentials

## Clean Up

Remove all resources:
```bash
# Delete Lambda function
aws lambda delete-function --function-name text-animation-processor

# Delete API Gateway
aws apigateway delete-rest-api --rest-api-id YOUR_API_ID

# Delete ECR repository
aws ecr delete-repository --repository-name text-animation-lambda --force

# Delete IAM role
aws iam detach-role-policy --role-name text-animation-processor-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role-policy --role-name text-animation-processor-role --policy-name S3Access
aws iam delete-role --role-name text-animation-processor-role

# Delete S3 bucket (after emptying it)
aws s3 rm s3://text-animation-videos --recursive
aws s3 rb s3://text-animation-videos
```

## Support

For issues or questions, please check:
- Lambda function logs in CloudWatch
- API Gateway execution logs
- S3 bucket permissions
- ECR repository status