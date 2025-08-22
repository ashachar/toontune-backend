/**
 * Monitoring and Metrics Utilities
 * CloudWatch metrics and X-Ray tracing integration
 */

const { cloudWatch } = require('./aws-clients');
const { logger } = require('./logger');

// Metric namespaces
const NAMESPACE = 'ToonTune/Lambda';
const CUSTOM_NAMESPACE = 'ToonTune/AI';

// Metric dimensions
const getDimensions = (functionName, tier = 'default') => [
  { Name: 'FunctionName', Value: functionName },
  { Name: 'Environment', Value: process.env.NODE_ENV || 'development' },
  { Name: 'Tier', Value: tier }
];

/**
 * Record function invocation metric
 */
async function recordInvocation(functionName, tier = 'default') {
  try {
    await cloudWatch.putMetricData({
      Namespace: NAMESPACE,
      MetricData: [
        {
          MetricName: 'Invocations',
          Value: 1,
          Unit: 'Count',
          Timestamp: new Date(),
          Dimensions: getDimensions(functionName, tier)
        }
      ]
    }).promise();
  } catch (error) {
    logger.error('Failed to record invocation metric', { error: error.message });
  }
}

/**
 * Record function duration metric
 */
async function recordDuration(functionName, duration, tier = 'default') {
  try {
    await cloudWatch.putMetricData({
      Namespace: NAMESPACE,
      MetricData: [
        {
          MetricName: 'Duration',
          Value: duration,
          Unit: 'Milliseconds',
          Timestamp: new Date(),
          Dimensions: getDimensions(functionName, tier)
        }
      ]
    }).promise();
  } catch (error) {
    logger.error('Failed to record duration metric', { error: error.message });
  }
}

/**
 * Record function error metric
 */
async function recordError(functionName, errorType, tier = 'default') {
  try {
    await cloudWatch.putMetricData({
      Namespace: NAMESPACE,
      MetricData: [
        {
          MetricName: 'Errors',
          Value: 1,
          Unit: 'Count',
          Timestamp: new Date(),
          Dimensions: [
            ...getDimensions(functionName, tier),
            { Name: 'ErrorType', Value: errorType }
          ]
        }
      ]
    }).promise();
  } catch (error) {
    logger.error('Failed to record error metric', { error: error.message });
  }
}

/**
 * Record AI service usage metrics
 */
async function recordAIUsage(service, operation, tokens = 0, cost = 0) {
  try {
    const metrics = [
      {
        MetricName: 'AIServiceCalls',
        Value: 1,
        Unit: 'Count',
        Dimensions: [
          { Name: 'Service', Value: service },
          { Name: 'Operation', Value: operation }
        ]
      }
    ];

    if (tokens > 0) {
      metrics.push({
        MetricName: 'TokensUsed',
        Value: tokens,
        Unit: 'Count',
        Dimensions: [
          { Name: 'Service', Value: service },
          { Name: 'Operation', Value: operation }
        ]
      });
    }

    if (cost > 0) {
      metrics.push({
        MetricName: 'EstimatedCost',
        Value: cost,
        Unit: 'None',
        Dimensions: [
          { Name: 'Service', Value: service },
          { Name: 'Operation', Value: operation }
        ]
      });
    }

    await cloudWatch.putMetricData({
      Namespace: CUSTOM_NAMESPACE,
      MetricData: metrics.map(m => ({ ...m, Timestamp: new Date() }))
    }).promise();
  } catch (error) {
    logger.error('Failed to record AI usage metric', { error: error.message });
  }
}

/**
 * Record cache hit/miss metrics
 */
async function recordCacheMetric(cacheName, hit) {
  try {
    await cloudWatch.putMetricData({
      Namespace: NAMESPACE,
      MetricData: [
        {
          MetricName: hit ? 'CacheHits' : 'CacheMisses',
          Value: 1,
          Unit: 'Count',
          Timestamp: new Date(),
          Dimensions: [
            { Name: 'CacheName', Value: cacheName },
            { Name: 'Environment', Value: process.env.NODE_ENV || 'development' }
          ]
        }
      ]
    }).promise();
  } catch (error) {
    logger.error('Failed to record cache metric', { error: error.message });
  }
}

/**
 * Record cold start metric
 */
async function recordColdStart(duration) {
  try {
    await cloudWatch.putMetricData({
      Namespace: NAMESPACE,
      MetricData: [
        {
          MetricName: 'ColdStarts',
          Value: 1,
          Unit: 'Count',
          Timestamp: new Date(),
          Dimensions: [
            { Name: 'Environment', Value: process.env.NODE_ENV || 'development' }
          ]
        },
        {
          MetricName: 'ColdStartDuration',
          Value: duration,
          Unit: 'Milliseconds',
          Timestamp: new Date(),
          Dimensions: [
            { Name: 'Environment', Value: process.env.NODE_ENV || 'development' }
          ]
        }
      ]
    }).promise();
  } catch (error) {
    logger.error('Failed to record cold start metric', { error: error.message });
  }
}

/**
 * Create CloudWatch alarm for error rate
 */
async function createErrorRateAlarm(functionName, threshold = 0.01) {
  try {
    const alarmName = `${functionName}-error-rate-alarm`;
    
    await cloudWatch.putMetricAlarm({
      AlarmName: alarmName,
      ComparisonOperator: 'GreaterThanThreshold',
      EvaluationPeriods: 2,
      MetricName: 'Errors',
      Namespace: NAMESPACE,
      Period: 300, // 5 minutes
      Statistic: 'Average',
      Threshold: threshold,
      ActionsEnabled: true,
      AlarmActions: process.env.SNS_TOPIC_ARN ? [process.env.SNS_TOPIC_ARN] : [],
      AlarmDescription: `Alarm when error rate exceeds ${threshold * 100}%`,
      Dimensions: getDimensions(functionName)
    }).promise();

    logger.info(`Created error rate alarm for ${functionName}`);
    return alarmName;
  } catch (error) {
    logger.error('Failed to create error rate alarm', { error: error.message });
    throw error;
  }
}

/**
 * Create cost anomaly detector
 */
async function createCostAnomalyDetector() {
  try {
    await cloudWatch.putAnomalyDetector({
      Namespace: CUSTOM_NAMESPACE,
      MetricName: 'EstimatedCost',
      Stat: 'Sum'
    }).promise();

    logger.info('Created cost anomaly detector');
  } catch (error) {
    logger.error('Failed to create cost anomaly detector', { error: error.message });
  }
}

/**
 * Track API usage per customer
 */
async function trackAPIUsage(apiKey, functionName, responseTime) {
  try {
    // Hash API key for privacy
    const crypto = require('crypto');
    const hashedKey = crypto.createHash('sha256').update(apiKey).digest('hex').substring(0, 16);

    await cloudWatch.putMetricData({
      Namespace: CUSTOM_NAMESPACE,
      MetricData: [
        {
          MetricName: 'APIUsage',
          Value: 1,
          Unit: 'Count',
          Timestamp: new Date(),
          Dimensions: [
            { Name: 'Customer', Value: hashedKey },
            { Name: 'Function', Value: functionName }
          ]
        },
        {
          MetricName: 'ResponseTime',
          Value: responseTime,
          Unit: 'Milliseconds',
          Timestamp: new Date(),
          Dimensions: [
            { Name: 'Customer', Value: hashedKey },
            { Name: 'Function', Value: functionName }
          ]
        }
      ]
    }).promise();
  } catch (error) {
    logger.error('Failed to track API usage', { error: error.message });
  }
}

/**
 * Initialize X-Ray tracing if enabled
 */
function initializeXRay() {
  if (process.env._X_AMZN_TRACE_ID) {
    try {
      const AWSXRay = require('aws-xray-sdk-core');
      const AWS = AWSXRay.captureAWS(require('aws-sdk'));
      
      // Capture HTTP calls
      AWSXRay.captureHTTPsGlobal(require('http'));
      AWSXRay.captureHTTPsGlobal(require('https'));
      
      // Add metadata
      const segment = AWSXRay.getSegment();
      if (segment) {
        segment.addAnnotation('service', 'toontune-lambda');
        segment.addAnnotation('environment', process.env.NODE_ENV || 'development');
      }
      
      logger.info('X-Ray tracing initialized');
      return AWSXRay;
    } catch (error) {
      logger.warn('Failed to initialize X-Ray tracing', { error: error.message });
    }
  }
  return null;
}

// Export metrics collector for batch operations
class MetricsCollector {
  constructor() {
    this.metrics = [];
    this.flushInterval = 10000; // 10 seconds
    this.maxBatchSize = 20;
    this.timer = null;
  }

  add(namespace, metricName, value, unit = 'Count', dimensions = []) {
    this.metrics.push({
      namespace,
      metricName,
      value,
      unit,
      dimensions,
      timestamp: new Date()
    });

    if (this.metrics.length >= this.maxBatchSize) {
      this.flush();
    } else if (!this.timer) {
      this.timer = setTimeout(() => this.flush(), this.flushInterval);
    }
  }

  async flush() {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }

    if (this.metrics.length === 0) return;

    // Group metrics by namespace
    const grouped = {};
    this.metrics.forEach(m => {
      if (!grouped[m.namespace]) {
        grouped[m.namespace] = [];
      }
      grouped[m.namespace].push({
        MetricName: m.metricName,
        Value: m.value,
        Unit: m.unit,
        Timestamp: m.timestamp,
        Dimensions: m.dimensions
      });
    });

    // Send metrics to CloudWatch
    const promises = Object.entries(grouped).map(([namespace, metricData]) =>
      cloudWatch.putMetricData({
        Namespace: namespace,
        MetricData: metricData
      }).promise().catch(error => {
        logger.error('Failed to flush metrics', { error: error.message, namespace });
      })
    );

    await Promise.all(promises);
    this.metrics = [];
  }
}

const metricsCollector = new MetricsCollector();

module.exports = {
  recordInvocation,
  recordDuration,
  recordError,
  recordAIUsage,
  recordCacheMetric,
  recordColdStart,
  createErrorRateAlarm,
  createCostAnomalyDetector,
  trackAPIUsage,
  initializeXRay,
  metricsCollector
};