const express = require('express');
const router = express.Router();
const DoodleGenerator = require('../doodle-generator');
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises;

// Initialize doodle generator
const doodleGen = new DoodleGenerator();

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadDir = path.join(__dirname, '..', '..', 'uploads', 'generated');
    await fs.mkdir(uploadDir, { recursive: true });
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'ai-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({ storage });

// Generate doodles from text description
router.post('/generate-doodle', async (req, res) => {
  const { description, style = 'simple', count = 4, provider = 'mock' } = req.body;

  if (!description) {
    return res.status(400).json({ error: 'Description is required' });
  }

  try {
    const result = await doodleGen.generateDoodleFromText(description, {
      style,
      count,
      provider
    });

    res.json({
      success: true,
      ...result
    });
  } catch (error) {
    console.error('Error generating doodle:', error);
    res.status(500).json({ 
      error: 'Failed to generate doodle',
      message: error.message 
    });
  }
});

// Generate animation sequence
router.post('/generate-animation', async (req, res) => {
  const { description, frameCount = 10 } = req.body;

  if (!description) {
    return res.status(400).json({ error: 'Description is required' });
  }

  try {
    const result = await doodleGen.generateAnimationSequence(description, frameCount);
    
    res.json({
      success: true,
      ...result
    });
  } catch (error) {
    console.error('Error generating animation:', error);
    res.status(500).json({ 
      error: 'Failed to generate animation sequence',
      message: error.message 
    });
  }
});

// Analyze doodle content
router.post('/analyze-doodle', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'Image file is required' });
  }

  try {
    const analysis = await doodleGen.analyzeDoodleContent(req.file.path);
    
    res.json({
      success: true,
      filename: req.file.filename,
      ...analysis
    });
  } catch (error) {
    console.error('Error analyzing doodle:', error);
    res.status(500).json({ 
      error: 'Failed to analyze doodle',
      message: error.message 
    });
  }
});

// Optimize doodle for canvas
router.post('/optimize-doodle', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'Image file is required' });
  }

  try {
    const optimized = await doodleGen.optimizeDoodleForCanvas(req.file.path);
    
    // Save optimized image
    const outputPath = req.file.path.replace(/\.[^/.]+$/, '-optimized.png');
    await fs.writeFile(outputPath, optimized.data);
    
    res.json({
      success: true,
      originalFile: req.file.filename,
      optimizedFile: path.basename(outputPath),
      ...optimized.dimensions
    });
  } catch (error) {
    console.error('Error optimizing doodle:', error);
    res.status(500).json({ 
      error: 'Failed to optimize doodle',
      message: error.message 
    });
  }
});

// Upload custom asset
router.post('/upload-asset', upload.single('asset'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  try {
    const assetUrl = `/uploads/generated/${req.file.filename}`;
    res.json({
      success: true,
      url: assetUrl,
      filename: req.file.filename,
      size: req.file.size
    });
  } catch (error) {
    console.error('Error uploading asset:', error);
    res.status(500).json({ error: 'Failed to upload asset' });
  }
});

module.exports = router;