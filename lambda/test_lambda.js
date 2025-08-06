/**
 * Test client for ToonTune.ai Lambda function
 */

const AWS = require('aws-sdk');
const fs = require('fs').promises;

// Configure AWS
AWS.config.update({
  region: process.env.AWS_REGION || 'us-east-1'
});

const lambda = new AWS.Lambda();
const FUNCTION_NAME = 'toontune-ai-service';

// Test functions
const tests = {
  // Test doodle generation
  async testGenerateDoodle() {
    console.log('Testing doodle generation...');
    
    const payload = {
      body: {
        action: 'generate-doodle',
        description: 'a cute robot dancing',
        style: 'simple',
        count: 2,
        provider: 'mock'
      }
    };

    const result = await invokeLambda(payload);
    console.log('Generate Doodle Result:', JSON.stringify(result, null, 2));
    return result;
  },

  // Test animation generation
  async testGenerateAnimation() {
    console.log('Testing animation generation...');
    
    const payload = {
      body: {
        action: 'generate-animation',
        description: 'a bird flying',
        frameCount: 5
      }
    };

    const result = await invokeLambda(payload);
    console.log('Generate Animation Result:', JSON.stringify(result, null, 2));
    return result;
  },

  // Test doodle analysis
  async testAnalyzeDoodle() {
    console.log('Testing doodle analysis...');
    
    // Create mock image data
    const mockImageData = Buffer.from('mock-image-data').toString('base64');
    
    const payload = {
      body: {
        action: 'analyze-doodle',
        imageData: mockImageData,
        mimeType: 'image/png'
      }
    };

    const result = await invokeLambda(payload);
    console.log('Analyze Doodle Result:', JSON.stringify(result, null, 2));
    return result;
  },

  // Test doodle optimization
  async testOptimizeDoodle() {
    console.log('Testing doodle optimization...');
    
    // Create mock image data
    const mockImageData = Buffer.from('mock-image-data').toString('base64');
    
    const payload = {
      body: {
        action: 'optimize-doodle',
        imageData: mockImageData,
        width: 1920,
        height: 1080
      }
    };

    const result = await invokeLambda(payload);
    console.log('Optimize Doodle Result:', JSON.stringify(result, null, 2));
    return result;
  },

  // Test warm-up
  async testWarmup() {
    console.log('Testing warm-up...');
    
    const payload = {
      warmup: true
    };

    const result = await invokeLambda(payload);
    console.log('Warm-up Result:', JSON.stringify(result, null, 2));
    return result;
  },

  // Test error handling
  async testErrorHandling() {
    console.log('Testing error handling...');
    
    const payload = {
      body: {
        action: 'invalid-action'
      }
    };

    const result = await invokeLambda(payload);
    console.log('Error Handling Result:', JSON.stringify(result, null, 2));
    return result;
  }
};

// Helper function to invoke Lambda
async function invokeLambda(payload) {
  const params = {
    FunctionName: FUNCTION_NAME,
    Payload: JSON.stringify(payload)
  };

  try {
    const response = await lambda.invoke(params).promise();
    const result = JSON.parse(response.Payload);
    
    if (response.StatusCode !== 200) {
      console.error('Lambda invocation failed:', response);
      return null;
    }

    return result;
  } catch (error) {
    console.error('Error invoking Lambda:', error);
    return null;
  }
}

// Run tests
async function runTests() {
  console.log('Starting Lambda function tests...\n');
  
  const testNames = Object.keys(tests);
  const results = {};
  
  for (const testName of testNames) {
    console.log(`\n${'='.repeat(50)}`);
    console.log(`Running: ${testName}`);
    console.log('='.repeat(50));
    
    try {
      const result = await tests[testName]();
      results[testName] = {
        success: result && result.statusCode === 200,
        result
      };
    } catch (error) {
      console.error(`Test ${testName} failed:`, error);
      results[testName] = {
        success: false,
        error: error.message
      };
    }
    
    console.log('\n');
  }
  
  // Print summary
  console.log('\n' + '='.repeat(50));
  console.log('TEST SUMMARY');
  console.log('='.repeat(50));
  
  let passed = 0;
  let failed = 0;
  
  for (const [testName, result] of Object.entries(results)) {
    if (result.success) {
      console.log(`✅ ${testName}: PASSED`);
      passed++;
    } else {
      console.log(`❌ ${testName}: FAILED`);
      failed++;
    }
  }
  
  console.log('\n' + '-'.repeat(50));
  console.log(`Total: ${passed} passed, ${failed} failed`);
  console.log('='.repeat(50));
  
  // Save results to file
  await fs.writeFile(
    'test_results.json',
    JSON.stringify(results, null, 2)
  );
  console.log('\nTest results saved to test_results.json');
}

// Run if executed directly
if (require.main === module) {
  runTests().catch(console.error);
}

module.exports = { tests, invokeLambda };