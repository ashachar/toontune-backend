/**
 * Comprehensive Unit Tests for ToonTune.ai Lambda Functions
 * Tests each function with various scenarios and edge cases
 */

const { handler } = require('../index');
const mockData = require('./mock-data');

// Test configuration
const TEST_TIMEOUT = 30000; // 30 seconds
const VERBOSE = process.env.VERBOSE === 'true';

// Color codes for console output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  gray: '\x1b[90m'
};

/**
 * Test runner utility
 */
class TestRunner {
  constructor() {
    this.tests = [];
    this.results = {
      passed: 0,
      failed: 0,
      skipped: 0,
      errors: []
    };
  }

  describe(suite, tests) {
    this.tests.push({ suite, tests });
  }

  async run() {
    console.log(`${colors.blue}Starting ToonTune.ai Lambda Unit Tests${colors.reset}\n`);
    const startTime = Date.now();

    for (const { suite, tests } of this.tests) {
      console.log(`${colors.yellow}Test Suite: ${suite}${colors.reset}`);
      
      for (const test of tests) {
        await this.runTest(suite, test);
      }
      
      console.log(''); // Empty line between suites
    }

    const duration = Date.now() - startTime;
    this.printSummary(duration);
  }

  async runTest(suite, test) {
    const { name, fn, skip = false } = test;
    
    if (skip) {
      console.log(`  ${colors.gray}⊘ ${name} (skipped)${colors.reset}`);
      this.results.skipped++;
      return;
    }

    try {
      const timeout = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Test timeout')), TEST_TIMEOUT)
      );
      
      const testExecution = fn();
      await Promise.race([testExecution, timeout]);
      
      console.log(`  ${colors.green}✓ ${name}${colors.reset}`);
      this.results.passed++;
    } catch (error) {
      console.log(`  ${colors.red}✗ ${name}${colors.reset}`);
      if (VERBOSE) {
        console.log(`    ${colors.red}Error: ${error.message}${colors.reset}`);
      }
      this.results.failed++;
      this.results.errors.push({
        suite,
        test: name,
        error: error.message
      });
    }
  }

  printSummary(duration) {
    const total = this.results.passed + this.results.failed + this.results.skipped;
    
    console.log(`${colors.blue}${'='.repeat(50)}${colors.reset}`);
    console.log(`${colors.blue}Test Summary${colors.reset}`);
    console.log(`${colors.blue}${'='.repeat(50)}${colors.reset}`);
    
    console.log(`Total Tests: ${total}`);
    console.log(`${colors.green}Passed: ${this.results.passed}${colors.reset}`);
    console.log(`${colors.red}Failed: ${this.results.failed}${colors.reset}`);
    console.log(`${colors.gray}Skipped: ${this.results.skipped}${colors.reset}`);
    console.log(`Duration: ${(duration / 1000).toFixed(2)}s`);
    
    if (this.results.errors.length > 0) {
      console.log(`\n${colors.red}Failed Tests:${colors.reset}`);
      this.results.errors.forEach(({ suite, test, error }) => {
        console.log(`  ${suite} > ${test}`);
        console.log(`    ${colors.gray}${error}${colors.reset}`);
      });
    }
    
    console.log(`${colors.blue}${'='.repeat(50)}${colors.reset}`);
    
    // Exit with appropriate code
    process.exit(this.results.failed > 0 ? 1 : 0);
  }
}

/**
 * Assert utilities
 */
const assert = {
  equal: (actual, expected, message) => {
    if (actual !== expected) {
      throw new Error(message || `Expected ${expected}, got ${actual}`);
    }
  },
  
  notEqual: (actual, expected, message) => {
    if (actual === expected) {
      throw new Error(message || `Expected not ${expected}, got ${actual}`);
    }
  },
  
  ok: (value, message) => {
    if (!value) {
      throw new Error(message || `Expected truthy value, got ${value}`);
    }
  },
  
  includes: (array, item, message) => {
    if (!array.includes(item)) {
      throw new Error(message || `Expected array to include ${item}`);
    }
  },
  
  hasKeys: (obj, keys, message) => {
    const missingKeys = keys.filter(key => !(key in obj));
    if (missingKeys.length > 0) {
      throw new Error(message || `Missing keys: ${missingKeys.join(', ')}`);
    }
  },
  
  isValidUrl: (url, message) => {
    try {
      new URL(url);
    } catch {
      throw new Error(message || `Invalid URL: ${url}`);
    }
  }
};

// Initialize test runner
const runner = new TestRunner();

// Test Suite: Authentication
runner.describe('Authentication', [
  {
    name: 'should reject requests without API key',
    fn: async () => {
      const event = mockData.generateMockLambdaEvent('generateScriptFromIdea', false);
      const context = mockData.generateMockLambdaContext();
      
      const response = await handler(event, context);
      const body = JSON.parse(response.body);
      
      assert.equal(response.statusCode, 401);
      assert.ok(body.error);
      assert.includes(body.error.toLowerCase(), 'api key');
    }
  },
  {
    name: 'should accept valid API key',
    fn: async () => {
      const event = mockData.generateMockLambdaEvent('generateScriptFromIdea', true);
      const context = mockData.generateMockLambdaContext();
      
      // Mock the actual function to avoid external API calls
      event.body = JSON.stringify({
        function: 'detectLanguage',
        text: 'Hello world'
      });
      
      const response = await handler(event, context);
      assert.equal(response.statusCode, 200);
    }
  },
  {
    name: 'should handle warm-up pings',
    fn: async () => {
      const event = { warmup: true };
      const context = mockData.generateMockLambdaContext();
      
      const response = await handler(event, context);
      const body = JSON.parse(response.body);
      
      assert.equal(response.statusCode, 200);
      assert.ok(body.message);
      assert.includes(body.message.toLowerCase(), 'warm');
    }
  }
]);

// Test Suite: Input Validation
runner.describe('Input Validation', [
  {
    name: 'should reject missing function name',
    fn: async () => {
      const event = {
        body: JSON.stringify({}),
        headers: { 'X-Api-Key': 'dev-key-12345' }
      };
      const context = mockData.generateMockLambdaContext();
      
      const response = await handler(event, context);
      const body = JSON.parse(response.body);
      
      assert.equal(response.statusCode, 400);
      assert.includes(body.error.toLowerCase(), 'function');
    }
  },
  {
    name: 'should reject unknown function',
    fn: async () => {
      const event = {
        body: JSON.stringify({ function: 'unknownFunction' }),
        headers: { 'X-Api-Key': 'dev-key-12345' }
      };
      const context = mockData.generateMockLambdaContext();
      
      const response = await handler(event, context);
      const body = JSON.parse(response.body);
      
      assert.equal(response.statusCode, 400);
      assert.includes(body.error.toLowerCase(), 'unknown');
    }
  },
  {
    name: 'should handle malformed JSON',
    fn: async () => {
      const event = {
        body: '{ invalid json }',
        headers: { 'X-Api-Key': 'dev-key-12345' }
      };
      const context = mockData.generateMockLambdaContext();
      
      const response = await handler(event, context);
      const body = JSON.parse(response.body);
      
      assert.equal(response.statusCode, 400);
      assert.ok(body.error);
    }
  }
]);

// Test Suite: Function Responses
runner.describe('Function Response Structure', [
  {
    name: 'should return proper response structure',
    fn: async () => {
      const event = {
        body: JSON.stringify({
          function: 'detectLanguage',
          text: 'Hello world'
        }),
        headers: { 'X-Api-Key': 'dev-key-12345' }
      };
      const context = mockData.generateMockLambdaContext();
      
      const response = await handler(event, context);
      const body = JSON.parse(response.body);
      
      assert.equal(response.statusCode, 200);
      assert.hasKeys(body, ['success', 'function', 'result', 'executionTime', 'requestId']);
      assert.equal(body.success, true);
      assert.equal(body.function, 'detectLanguage');
      assert.ok(body.executionTime > 0);
    }
  },
  {
    name: 'should include CORS headers',
    fn: async () => {
      const event = {
        body: JSON.stringify({
          function: 'detectLanguage',
          text: 'Test'
        }),
        headers: { 'X-Api-Key': 'dev-key-12345' }
      };
      const context = mockData.generateMockLambdaContext();
      
      const response = await handler(event, context);
      
      assert.ok(response.headers);
      assert.equal(response.headers['Access-Control-Allow-Origin'], '*');
      assert.ok(response.headers['Access-Control-Allow-Headers']);
    }
  }
]);

// Test Suite: Error Handling
runner.describe('Error Handling', [
  {
    name: 'should format errors properly',
    fn: async () => {
      const event = {
        body: JSON.stringify({
          function: 'generateScriptFromIdea'
          // Missing required 'idea' field
        }),
        headers: { 'X-Api-Key': 'dev-key-12345' }
      };
      const context = mockData.generateMockLambdaContext();
      
      const response = await handler(event, context);
      const body = JSON.parse(response.body);
      
      assert.equal(response.statusCode, 400);
      assert.equal(body.success, false);
      assert.ok(body.error);
      assert.ok(body.requestId);
    }
  },
  {
    name: 'should indicate retryable errors',
    fn: async () => {
      // This would need to mock a network error
      // Skipping for now as it requires more complex mocking
      assert.ok(true);
    },
    skip: true
  }
]);

// Test Suite: Language Detection
runner.describe('Language Detection Function', [
  {
    name: 'should detect English text',
    fn: async () => {
      const event = {
        body: JSON.stringify({
          function: 'detectLanguage',
          text: 'This is a test in English'
        }),
        headers: { 'X-Api-Key': 'dev-key-12345' }
      };
      const context = mockData.generateMockLambdaContext();
      
      const response = await handler(event, context);
      const body = JSON.parse(response.body);
      
      assert.equal(response.statusCode, 200);
      assert.ok(body.result.primaryLanguage);
      assert.equal(body.result.primaryLanguage.code, 'en');
    }
  },
  {
    name: 'should handle empty text',
    fn: async () => {
      const event = {
        body: JSON.stringify({
          function: 'detectLanguage',
          text: ''
        }),
        headers: { 'X-Api-Key': 'dev-key-12345' }
      };
      const context = mockData.generateMockLambdaContext();
      
      const response = await handler(event, context);
      const body = JSON.parse(response.body);
      
      assert.equal(response.statusCode, 200);
      assert.ok(body.result.primaryLanguage);
      assert.equal(body.result.primaryLanguage.code, 'unknown');
    }
  }
]);

// Test Suite: Backward Compatibility
runner.describe('Backward Compatibility', [
  {
    name: 'should support legacy generate-doodle function',
    fn: async () => {
      const event = {
        body: JSON.stringify({
          function: 'generate-doodle',
          prompt: 'test prompt'
        }),
        headers: { 'X-Api-Key': 'dev-key-12345' }
      };
      const context = mockData.generateMockLambdaContext();
      
      const response = await handler(event, context);
      assert.equal(response.statusCode, 200);
    }
  },
  {
    name: 'should support action field instead of function',
    fn: async () => {
      const event = {
        body: JSON.stringify({
          action: 'detectLanguage',
          text: 'Test'
        }),
        headers: { 'X-Api-Key': 'dev-key-12345' }
      };
      const context = mockData.generateMockLambdaContext();
      
      const response = await handler(event, context);
      const body = JSON.parse(response.body);
      
      assert.equal(response.statusCode, 200);
      assert.equal(body.function, 'detectLanguage');
    }
  }
]);

// Test Suite: Performance
runner.describe('Performance', [
  {
    name: 'should complete simple requests quickly',
    fn: async () => {
      const event = {
        body: JSON.stringify({
          function: 'detectLanguage',
          text: 'Quick test'
        }),
        headers: { 'X-Api-Key': 'dev-key-12345' }
      };
      const context = mockData.generateMockLambdaContext();
      
      const startTime = Date.now();
      const response = await handler(event, context);
      const duration = Date.now() - startTime;
      const body = JSON.parse(response.body);
      
      assert.equal(response.statusCode, 200);
      assert.ok(duration < 1000, `Request took ${duration}ms, expected < 1000ms`);
      assert.ok(body.executionTime < 1000);
    }
  }
]);

// Test Suite: Multi-language Support
runner.describe('Multi-language Support', [
  {
    name: 'should handle Spanish text',
    fn: async () => {
      const event = {
        body: JSON.stringify({
          function: 'detectLanguage',
          text: 'Hola, esto es una prueba en español'
        }),
        headers: { 'X-Api-Key': 'dev-key-12345' }
      };
      const context = mockData.generateMockLambdaContext();
      
      const response = await handler(event, context);
      const body = JSON.parse(response.body);
      
      assert.equal(response.statusCode, 200);
      assert.ok(body.result.primaryLanguage);
      assert.includes(['es', 'spanish'], body.result.primaryLanguage.code.toLowerCase());
    }
  },
  {
    name: 'should handle mixed language content',
    fn: async () => {
      const event = {
        body: JSON.stringify({
          function: 'detectLanguage',
          text: 'Hello world. Bonjour le monde. Hola mundo.'
        }),
        headers: { 'X-Api-Key': 'dev-key-12345' }
      };
      const context = mockData.generateMockLambdaContext();
      
      const response = await handler(event, context);
      const body = JSON.parse(response.body);
      
      assert.equal(response.statusCode, 200);
      assert.ok(body.result.primaryLanguage);
      assert.ok(body.result.alternatives || body.result.mixedContent);
    }
  }
]);

// Run all tests
(async () => {
  try {
    await runner.run();
  } catch (error) {
    console.error(`${colors.red}Test runner failed: ${error.message}${colors.reset}`);
    process.exit(1);
  }
})();