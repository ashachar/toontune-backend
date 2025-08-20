#!/usr/bin/env node

/**
 * Test script for text animation Lambda function
 */

const fs = require('fs');
const path = require('path');

// Mock AWS SDK
const mockS3 = {
    upload: jest.fn().mockImplementation((params, callback) => {
        return {
            promise: () => Promise.resolve({
                Location: `https://test-bucket.s3.us-east-1.amazonaws.com/${params.Key}`
            })
        };
    }),
    getObject: jest.fn().mockImplementation((params) => {
        return {
            promise: () => Promise.resolve({
                Body: Buffer.from('mock video data')
            })
        };
    })
};

// Mock AWS SDK
jest.mock('aws-sdk', () => ({
    S3: jest.fn(() => mockS3)
}));

// Import the function after mocking
const processTextAnimation = require('./functions/processTextAnimation');

async function testLocal() {
    console.log('🧪 Testing Text Animation Lambda Function\n');
    
    // Test event
    const testEvent = {
        video_url: 'uploads/assets/videos/do_re_mi.mov',
        text: 'HELLO',
        style: 'default'
    };
    
    console.log('📋 Test Input:');
    console.log(JSON.stringify(testEvent, null, 2));
    console.log();
    
    try {
        // Call the function
        console.log('🎬 Processing video...');
        const result = await processTextAnimation(testEvent);
        
        console.log('\n✅ Success! Result:');
        console.log(JSON.stringify(result, null, 2));
        
        if (result.statusCode === 200) {
            const body = JSON.parse(result.body);
            console.log('\n📊 Summary:');
            console.log(`  - Output URL: ${body.data.video_url}`);
            console.log(`  - Processing Time: ${body.data.processing_time}`);
            console.log(`  - Output Size: ${body.data.output_size}`);
        }
        
    } catch (error) {
        console.error('\n❌ Error:', error.message);
        console.error(error.stack);
    }
}

// Direct Python test (bypassing Node.js wrapper)
async function testPythonDirectly() {
    console.log('\n🐍 Testing Python processor directly...\n');
    
    const { spawn } = require('child_process');
    const inputVideo = 'uploads/assets/videos/do_re_mi.mov';
    const outputVideo = '/tmp/test_animation_output.mp4';
    const text = 'START';
    
    return new Promise((resolve, reject) => {
        const pythonPath = path.join(__dirname, 'python', 'text_animation_processor.py');
        
        console.log(`Running: python3 ${pythonPath} ${inputVideo} ${text} ${outputVideo}`);
        
        const python = spawn('python3', [
            pythonPath,
            inputVideo,
            text,
            outputVideo
        ]);
        
        python.stdout.on('data', (data) => {
            console.log(`Python: ${data}`);
        });
        
        python.stderr.on('data', (data) => {
            console.error(`Python Error: ${data}`);
        });
        
        python.on('close', (code) => {
            if (code === 0) {
                console.log(`\n✅ Python processing complete!`);
                console.log(`   Output: ${outputVideo}`);
                
                // Check if file exists
                if (fs.existsSync(outputVideo)) {
                    const stats = fs.statSync(outputVideo);
                    const sizeMB = stats.size / (1024 * 1024);
                    console.log(`   Size: ${sizeMB.toFixed(2)} MB`);
                    
                    // Create GIF for preview
                    const gifOutput = outputVideo.replace('.mp4', '.gif');
                    const { execSync } = require('child_process');
                    
                    try {
                        execSync(`ffmpeg -i ${outputVideo} -vf "fps=10,scale=400:-1" ${gifOutput} -y`);
                        console.log(`   GIF: ${gifOutput}`);
                    } catch (err) {
                        console.warn('   Could not create GIF preview');
                    }
                }
                
                resolve();
            } else {
                reject(new Error(`Python process exited with code ${code}`));
            }
        });
    });
}

// Run tests
async function runTests() {
    try {
        // First test Python directly
        await testPythonDirectly();
        
        // Then test the full Lambda function
        // Note: This requires proper mocking setup
        // await testLocal();
        
    } catch (error) {
        console.error('Test failed:', error);
        process.exit(1);
    }
}

// Run if called directly
if (require.main === module) {
    runTests();
}