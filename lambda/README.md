# ToonTune.ai Lambda Functions

AWS Lambda implementation for ToonTune.ai services, providing serverless doodle generation, analysis, and optimization.

## Features

- **Doodle Generation**: Generate AI-powered doodles from text descriptions
- **Animation Sequences**: Create animated doodle sequences
- **Doodle Analysis**: Analyze doodle content using AI
- **Image Optimization**: Optimize doodles for canvas rendering
- **Multiple AI Providers**: Support for OpenAI, Replicate, and Google Gemini
- **Warm Start Optimization**: Reduced cold start latency

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   API Gateway   │────▶│    Lambda    │────▶│ AI Services │
└─────────────────┘     └──────────────┘     └─────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  S3 Storage  │
                        └──────────────┘
```

## Setup

### Prerequisites

1. AWS CLI installed and configured
2. Node.js 18.x or later
3. AWS account with appropriate permissions

### Environment Variables

Create a `.env` file in the backend directory with:

```bash
# AI Service Keys
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
REPLICATE_API_KEY=your_replicate_key

# AWS Configuration
AWS_REGION=us-east-1
```

### Deployment

1. Make the deployment script executable:
```bash
chmod +x deploy.sh
chmod +x create_api_gateway.sh
```

2. Deploy the Lambda function:
```bash
./deploy.sh
```

3. Create API Gateway (optional):
```bash
./create_api_gateway.sh
```

## API Endpoints

### Generate Doodle

```javascript
POST /ai
{
  "action": "generate-doodle",
  "description": "a cute cat playing piano",
  "style": "simple",  // simple, detailed, cartoon, minimalist
  "count": 4,
  "provider": "mock"  // mock, openai, replicate
}
```

### Generate Animation

```javascript
POST /ai
{
  "action": "generate-animation",
  "description": "a bird flying",
  "frameCount": 10
}
```

### Analyze Doodle

```javascript
POST /ai
{
  "action": "analyze-doodle",
  "imageData": "base64_encoded_image_data",
  "mimeType": "image/png"
}
```

### Optimize Doodle

```javascript
POST /ai
{
  "action": "optimize-doodle",
  "imageData": "base64_encoded_image_data",
  "width": 1920,
  "height": 1080
}
```

## Testing

### Local Testing

Test the Lambda function locally:

```bash
node test_lambda.js
```

### AWS Testing

Test the deployed Lambda:

```bash
aws lambda invoke \
  --function-name toontune-ai-service \
  --payload file://test_payload.json \
  response.json
```

### API Gateway Testing

Test via API Gateway:

```bash
curl -X POST https://your-api-gateway-url/prod/ai \
  -H 'Content-Type: application/json' \
  -d '{"action":"generate-doodle","description":"cute cat","provider":"mock"}'
```

## Performance Optimization

### Cold Start Mitigation

1. **Warm-up Events**: The Lambda handles warm-up pings to keep the function warm
2. **Lazy Loading**: Providers are initialized only when needed
3. **Memory Allocation**: Set to 2GB for optimal performance
4. **Package Optimization**: Minimal dependencies and optimized bundles

### Best Practices

- Use environment variables for sensitive data
- Implement proper error handling
- Monitor CloudWatch logs for debugging
- Set up CloudWatch alarms for errors
- Use X-Ray for tracing (optional)

## Monitoring

### CloudWatch Logs

View Lambda logs:

```bash
aws logs tail /aws/lambda/toontune-ai-service --follow
```

### Metrics

Monitor key metrics:
- Invocation count
- Error rate
- Duration
- Concurrent executions
- Cold starts

## Cost Optimization

1. **Request Batching**: Combine multiple operations when possible
2. **Caching**: Implement caching for frequently accessed data
3. **Reserved Concurrency**: Set limits to control costs
4. **Provisioned Concurrency**: For predictable traffic patterns

## Security

1. **IAM Roles**: Minimal permissions principle
2. **API Keys**: Store in environment variables
3. **CORS**: Configure based on your frontend domain
4. **Rate Limiting**: Implement via API Gateway
5. **Input Validation**: Validate all incoming requests

## Troubleshooting

### Common Issues

1. **Timeout Errors**
   - Increase timeout in deploy.sh
   - Optimize image processing

2. **Memory Errors**
   - Increase memory allocation
   - Optimize image sizes

3. **Permission Errors**
   - Check IAM role permissions
   - Verify API keys

4. **Cold Start Latency**
   - Implement warm-up strategy
   - Use provisioned concurrency

## Development

### Local Development

1. Install dependencies:
```bash
npm install
```

2. Test locally:
```bash
node test_lambda.js
```

### Adding New Features

1. Update `lambda_function.js` with new action handlers
2. Update `doodle_generator_lambda.js` with new methods
3. Add test cases in `test_lambda.js`
4. Update deployment if new dependencies added

## Deployment Pipeline

### Manual Deployment

```bash
./deploy.sh
```

### CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Deploy Lambda

on:
  push:
    branches: [main]
    paths:
      - 'backend/lambda/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      - name: Deploy Lambda
        run: |
          cd backend/lambda
          ./deploy.sh
```

## Support

For issues or questions:
1. Check CloudWatch logs
2. Review this documentation
3. Test with mock provider first
4. Verify environment variables

## License

MIT