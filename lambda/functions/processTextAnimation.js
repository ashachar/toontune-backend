/**
 * Process Text Animation Lambda Function
 * Applies text animation effects (shrink → behind → dissolve) to videos
 */

const AWS = require('aws-sdk');
const { spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const axios = require('axios');
const { createSuccessResponse, createErrorResponse } = require('../utils/response');
const { validateInput } = require('../utils/validation');
const logger = require('../utils/logger');

const s3 = new AWS.S3();
const BUCKET_NAME = process.env.S3_BUCKET || 'toontune-text-animations';
const REGION = process.env.AWS_REGION || 'us-east-1';
const PYTHON_SCRIPT_PATH = path.join(__dirname, '..', 'python', 'text_animation_processor.py');

/**
 * Download video from URL to local file
 */
async function downloadVideo(videoUrl) {
    const tempPath = `/tmp/input_${uuidv4()}.mp4`;
    
    if (videoUrl.startsWith('s3://')) {
        // Parse S3 URL
        const match = videoUrl.match(/^s3:\/\/([^\/]+)\/(.+)$/);
        if (!match) throw new Error('Invalid S3 URL format');
        
        const [, bucket, key] = match;
        const params = { Bucket: bucket, Key: key };
        
        const data = await s3.getObject(params).promise();
        await fs.writeFile(tempPath, data.Body);
    } else {
        // Download from HTTP/HTTPS
        const response = await axios.get(videoUrl, {
            responseType: 'stream',
            timeout: 30000
        });
        
        const writer = fs.createWriteStream(tempPath);
        response.data.pipe(writer);
        
        await new Promise((resolve, reject) => {
            writer.on('finish', resolve);
            writer.on('error', reject);
        });
    }
    
    return tempPath;
}

/**
 * Upload processed video to S3
 */
async function uploadToS3(localPath, key) {
    const fileContent = await fs.readFile(localPath);
    
    const params = {
        Bucket: BUCKET_NAME,
        Key: key,
        Body: fileContent,
        ContentType: 'video/mp4',
        ACL: 'public-read'
    };
    
    await s3.upload(params).promise();
    
    return `https://${BUCKET_NAME}.s3.${REGION}.amazonaws.com/${key}`;
}

/**
 * Process video with Python script
 */
async function processVideoWithPython(inputPath, text, outputPath) {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python3', [
            PYTHON_SCRIPT_PATH,
            inputPath,
            text,
            outputPath
        ]);
        
        let stdout = '';
        let stderr = '';
        
        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
            logger.info('Python stdout:', data.toString());
        });
        
        pythonProcess.stderr.on('data', (data) => {
            stderr += data.toString();
            logger.error('Python stderr:', data.toString());
        });
        
        pythonProcess.on('close', (code) => {
            if (code === 0) {
                resolve({ stdout, stderr });
            } else {
                reject(new Error(`Python process exited with code ${code}: ${stderr}`));
            }
        });
        
        pythonProcess.on('error', (err) => {
            reject(new Error(`Failed to start Python process: ${err.message}`));
        });
    });
}

/**
 * Main handler for text animation processing
 */
async function processTextAnimation(event) {
    const startTime = Date.now();
    let inputPath = null;
    let outputPath = null;
    
    try {
        // Validate input
        const schema = {
            video_url: { type: 'string', required: true },
            text: { type: 'string', required: true, maxLength: 50 },
            style: { type: 'string', required: false }
        };
        
        const validationResult = validateInput(event, schema);
        if (!validationResult.valid) {
            return createErrorResponse(400, 'Invalid input', validationResult.errors);
        }
        
        const { video_url, text, style = 'default' } = event;
        const processedText = text.toUpperCase();
        
        logger.info('Processing text animation', {
            videoUrl: video_url,
            text: processedText,
            style
        });
        
        // Download input video
        logger.info('Downloading video...');
        inputPath = await downloadVideo(video_url);
        const inputSize = (await fs.stat(inputPath)).size / (1024 * 1024);
        logger.info(`Downloaded video: ${inputSize.toFixed(2)} MB`);
        
        // Process video
        outputPath = `/tmp/output_${uuidv4()}.mp4`;
        logger.info('Processing video with text animation...');
        
        await processVideoWithPython(inputPath, processedText, outputPath);
        
        // Verify output exists
        const outputStats = await fs.stat(outputPath);
        const outputSize = outputStats.size / (1024 * 1024);
        logger.info(`Processed video: ${outputSize.toFixed(2)} MB`);
        
        // Upload to S3
        const outputKey = `processed/${uuidv4()}.mp4`;
        logger.info(`Uploading to S3: ${outputKey}`);
        const outputUrl = await uploadToS3(outputPath, outputKey);
        
        // Calculate processing time
        const processingTime = (Date.now() - startTime) / 1000;
        
        // Return success response
        return createSuccessResponse({
            video_url: outputUrl,
            text: processedText,
            processing_time: `${processingTime.toFixed(2)}s`,
            output_size: `${outputSize.toFixed(2)} MB`,
            s3_key: outputKey
        });
        
    } catch (error) {
        logger.error('Error processing text animation:', error);
        
        if (error.message.includes('video_url')) {
            return createErrorResponse(400, 'Invalid video URL', { error: error.message });
        }
        
        if (error.message.includes('Python')) {
            return createErrorResponse(500, 'Video processing failed', { error: error.message });
        }
        
        return createErrorResponse(500, 'Internal server error', { error: error.message });
        
    } finally {
        // Cleanup temp files
        if (inputPath) {
            try {
                await fs.unlink(inputPath);
            } catch (err) {
                logger.warn('Failed to delete input file:', err);
            }
        }
        
        if (outputPath) {
            try {
                await fs.unlink(outputPath);
            } catch (err) {
                logger.warn('Failed to delete output file:', err);
            }
        }
    }
}

module.exports = processTextAnimation;