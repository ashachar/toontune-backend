/**
 * Comprehensive Test Suite for ToonTune.ai Lambda Functions
 * Tests all implemented functions with sample data
 */

const AWS = require('aws-sdk');
const fs = require('fs').promises;

// Configure AWS
AWS.config.update({
  region: process.env.AWS_REGION || 'us-east-1'
});

const lambda = new AWS.Lambda();
const FUNCTION_NAME = 'toontune-ai-service';

// Test data
const TEST_DATA = {
  generateScriptFromIdea: {
    function: 'generateScriptFromIdea',
    idea: 'The importance of recycling and protecting our environment',
    targetLength: 'short',
    language: 'en',
    tone: 'educational'
  },

  parseScriptToSlides: {
    function: 'parseScriptToSlides',
    script: 'Welcome to our journey about recycling. Every day, we produce waste that affects our planet. Recycling helps reduce pollution and saves natural resources. When we recycle paper, we save trees. When we recycle plastic, we reduce ocean pollution. Start small by sorting your waste at home. Together, we can make a big difference for future generations.',
    maxSlidesCount: 5,
    targetDuration: 5
  },

  generateSlideImages: {
    function: 'generateSlideImages',
    slides: [
      {
        slideNumber: 1,
        imagePrompt: 'Earth surrounded by recycling symbols',
        text: 'Welcome to recycling'
      },
      {
        slideNumber: 2,
        imagePrompt: 'Sorting bins for different materials',
        text: 'Sort your waste'
      }
    ],
    projectStyle: 'simple',
    batchSize: 2
  },

  generateVoiceover: {
    function: 'generateVoiceover',
    text: 'Welcome to ToonTune AI. This is a test of our voice generation system.',
    voiceId: 'delilah',
    language: 'en',
    speed: 1.0,
    pitch: 1.0
  },

  previewVoice: {
    function: 'previewVoice',
    voiceId: 'delilah',
    language: 'en'
  },

  syncVoiceToSlides: {
    function: 'syncVoiceToSlides',
    slides: [
      {
        text: 'First slide text',
        audioUrl: 'https://example.com/audio1.mp3',
        duration: 3.5
      },
      {
        text: 'Second slide text',
        audioUrl: 'https://example.com/audio2.mp3',
        duration: 4.2
      }
    ],
    transitionDuration: 0.5
  },

  detectLanguage: {
    function: 'detectLanguage',
    text: 'This is a sample text in English to detect the language.'
  },

  generateDoodleAsset: {
    function: 'generateDoodleAsset',
    description: 'A friendly robot waving hello',
    category: 'character',
    style: 'simple',
    variations: 2
  },

  applyStyleToProject: {
    function: 'applyStyleToProject',
    projectId: 'test-project-123',
    style: 'funky',
    elements: ['images', 'transitions', 'animations']
  },

  batchProcessProject: {
    function: 'batchProcessProject',
    projectData: {
      idea: 'The benefits of daily exercise',
      style: 'simple',
      voiceId: 'zachary',
      language: 'en'
    },
    options: {
      generateImages: true,
      generateVoiceover: true,
      autoSync: true,
      maxSlides: 3
    }
  }
};

// Test runner
async function runTest(testName, testData) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Testing: ${testName}`);
  console.log('='.repeat(60));
  
  const startTime = Date.now();
  
  try {
    const payload = {
      body: JSON.stringify(testData)
    };

    const params = {
      FunctionName: FUNCTION_NAME,
      Payload: JSON.stringify(payload)
    };

    const response = await lambda.invoke(params).promise();
    const result = JSON.parse(response.Payload);
    
    const duration = Date.now() - startTime;

    if (response.StatusCode === 200 && result.statusCode === 200) {
      const body = JSON.parse(result.body);
      
      console.log(`✅ SUCCESS (${duration}ms)`);
      console.log('Response:', JSON.stringify(body, null, 2).substring(0, 500) + '...');
      
      return {
        test: testName,
        success: true,
        duration,
        result: body
      };
    } else {
      console.log(`❌ FAILED (${duration}ms)`);
      console.log('Error:', result);
      
      return {
        test: testName,
        success: false,
        duration,
        error: result
      };
    }
  } catch (error) {
    const duration = Date.now() - startTime;
    console.log(`❌ ERROR (${duration}ms)`);
    console.log('Error:', error.message);
    
    return {
      test: testName,
      success: false,
      duration,
      error: error.message
    };
  }
}

// Main test execution
async function runAllTests() {
  console.log('🚀 Starting ToonTune.ai Lambda Function Tests');
  console.log('='.repeat(60));
  
  const results = [];
  
  // Run tests in sequence to avoid rate limiting
  for (const [testName, testData] of Object.entries(TEST_DATA)) {
    const result = await runTest(testName, testData);
    results.push(result);
    
    // Small delay between tests
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  // Print summary
  console.log('\n' + '='.repeat(60));
  console.log('📊 TEST SUMMARY');
  console.log('='.repeat(60));
  
  const successful = results.filter(r => r.success).length;
  const failed = results.filter(r => !r.success).length;
  const totalDuration = results.reduce((sum, r) => sum + r.duration, 0);
  
  results.forEach(result => {
    const icon = result.success ? '✅' : '❌';
    console.log(`${icon} ${result.test}: ${result.success ? 'PASSED' : 'FAILED'} (${result.duration}ms)`);
  });
  
  console.log('\n' + '-'.repeat(60));
  console.log(`Total Tests: ${results.length}`);
  console.log(`Passed: ${successful}`);
  console.log(`Failed: ${failed}`);
  console.log(`Success Rate: ${((successful / results.length) * 100).toFixed(1)}%`);
  console.log(`Total Duration: ${totalDuration}ms`);
  console.log('='.repeat(60));
  
  // Save results to file
  const report = {
    timestamp: new Date().toISOString(),
    functionName: FUNCTION_NAME,
    summary: {
      total: results.length,
      passed: successful,
      failed,
      successRate: (successful / results.length) * 100,
      totalDuration
    },
    results
  };
  
  await fs.writeFile(
    'test_results_comprehensive.json',
    JSON.stringify(report, null, 2)
  );
  
  console.log('\n📝 Detailed results saved to test_results_comprehensive.json');
}

// Run individual test (for debugging)
async function runSingleTest(functionName) {
  const testData = TEST_DATA[functionName];
  
  if (!testData) {
    console.error(`Test not found: ${functionName}`);
    console.log('Available tests:', Object.keys(TEST_DATA).join(', '));
    return;
  }
  
  await runTest(functionName, testData);
}

// CLI interface
if (require.main === module) {
  const args = process.argv.slice(2);
  
  if (args.length > 0) {
    // Run specific test
    runSingleTest(args[0]).catch(console.error);
  } else {
    // Run all tests
    runAllTests().catch(console.error);
  }
}

module.exports = {
  TEST_DATA,
  runTest,
  runAllTests,
  runSingleTest
};