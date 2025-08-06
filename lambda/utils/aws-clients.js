/**
 * AWS SDK Client Initialization
 * Centralized AWS service clients with proper configuration
 */

const AWS = require('aws-sdk');

// Configure AWS region
AWS.config.update({
  region: process.env.AWS_REGION || 'us-east-1'
});

// S3 Client
const s3 = new AWS.S3({
  apiVersion: '2006-03-01',
  signatureVersion: 'v4'
});

// Polly Client for Text-to-Speech
const polly = new AWS.Polly({
  apiVersion: '2016-06-10'
});

// Comprehend Client for Language Detection
const comprehend = new AWS.Comprehend({
  apiVersion: '2017-11-27'
});

// Translate Client for Multi-language Support
const translate = new AWS.Translate({
  apiVersion: '2017-07-01'
});

// Step Functions Client for Orchestration
const stepFunctions = new AWS.StepFunctions({
  apiVersion: '2016-11-23'
});

// CloudWatch Client for Metrics
const cloudWatch = new AWS.CloudWatch({
  apiVersion: '2010-08-01'
});

// Helper functions
const awsHelpers = {
  /**
   * Generate a presigned URL for S3 object
   */
  async getPresignedUrl(bucket, key, expiresIn = 3600) {
    const params = {
      Bucket: bucket || process.env.S3_BUCKET_NAME,
      Key: key,
      Expires: expiresIn
    };
    return s3.getSignedUrlPromise('getObject', params);
  },

  /**
   * Upload data to S3
   */
  async uploadToS3(bucket, key, data, contentType = 'application/octet-stream', metadata = {}) {
    const params = {
      Bucket: bucket || process.env.S3_BUCKET_NAME,
      Key: key,
      Body: data,
      ContentType: contentType,
      Metadata: metadata
    };
    
    const result = await s3.upload(params).promise();
    return result.Location;
  },

  /**
   * Check if S3 object exists
   */
  async s3ObjectExists(bucket, key) {
    try {
      await s3.headObject({
        Bucket: bucket || process.env.S3_BUCKET_NAME,
        Key: key
      }).promise();
      return true;
    } catch (error) {
      if (error.code === 'NotFound') {
        return false;
      }
      throw error;
    }
  },

  /**
   * Send custom metric to CloudWatch
   */
  async putMetric(namespace, metricName, value, unit = 'Count') {
    const params = {
      Namespace: namespace || 'ToonTune/Lambda',
      MetricData: [
        {
          MetricName: metricName,
          Value: value,
          Unit: unit,
          Timestamp: new Date()
        }
      ]
    };
    
    return cloudWatch.putMetricData(params).promise();
  }
};

module.exports = {
  s3,
  polly,
  comprehend,
  translate,
  stepFunctions,
  cloudWatch,
  awsHelpers
};