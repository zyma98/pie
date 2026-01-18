#!/usr/bin/env node

/**
 * Fetch and parse model metadata from pie-project/model-index repository
 * This script runs during build to generate src/data/models.json
 */

const fs = require('fs');
const path = require('path');
const toml = require('@iarna/toml');

const GITHUB_REPO = 'pie-project/model-index';
const GITHUB_API_URL = `https://api.github.com/repos/${GITHUB_REPO}/contents`;
const GITHUB_RAW_URL = `https://raw.githubusercontent.com/${GITHUB_REPO}/refs/heads/main`;
const OUTPUT_FILE = path.join(__dirname, '../src/data/models.json');

/**
 * Fetch data from URL
 */
async function fetchData(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return response;
}

/**
 * Extract parameter size from model name (e.g., "qwen-2.5-14b-instruct" -> "14B")
 */
function extractParameterSize(modelName) {
  const match = modelName.match(/(\d+(?:\.\d+)?)(b|B)(?:-|$)/);
  if (match) {
    const size = parseFloat(match[1]);
    return size < 1 ? `${size * 1000}M` : `${size}B`;
  }
  return null;
}

/**
 * Detect if model is a vision model from TOML structure
 */
function isVisionModel(tomlData) {
  // Check if architecture has vision section
  if (tomlData.architecture && tomlData.architecture.vision) {
    return true;
  }

  // Check if architecture type contains 'vl'
  if (tomlData.architecture && tomlData.architecture.type) {
    const type = tomlData.architecture.type.toLowerCase();
    if (type.includes('vl') || type.includes('vision')) {
      return true;
    }
  }

  return false;
}

/**
 * Extract Hugging Face URL from source section
 */
function extractHuggingFaceUrl(tomlData) {
  if (!tomlData.source) {
    return null;
  }

  // Get first URL from source section
  const sourceValues = Object.values(tomlData.source);
  for (const value of sourceValues) {
    if (typeof value === 'string' && value.includes('huggingface.co')) {
      // Extract base model URL (remove /resolve/main/... part)
      const match = value.match(/(https:\/\/huggingface\.co\/[^/]+\/[^/]+)/);
      return match ? match[1] : null;
    }
  }

  return null;
}

/**
 * Parse model tags from various fields
 */
function extractTags(modelName, tomlData, isVision) {
  const tags = [];

  if (isVision) {
    tags.push('vision');
  }

  if (modelName.includes('instruct')) {
    tags.push('instruct');
  }

  // Check for reasoning/thinking models
  if (modelName.includes('r1') || modelName.toLowerCase().includes('reasoning')) {
    tags.push('reasoning');
  }

  return tags;
}

/**
 * Main function to fetch and process models
 */
async function fetchModels() {
  console.log('🔍 Fetching model list from GitHub...');

  try {
    // 1. Fetch list of files from GitHub API
    const filesResponse = await fetchData(GITHUB_API_URL);
    const files = await filesResponse.json();

    // 2. Filter for .toml files (excluding traits.toml)
    const tomlFiles = files.filter(file =>
      file.name.endsWith('.toml') &&
      file.name !== 'traits.toml'
    );

    console.log(`📦 Found ${tomlFiles.length} model files`);

    // 3. Fetch platform support data
    let platformSupport = {};
    try {
      const platformResponse = await fetchData(`${GITHUB_RAW_URL}/platform-support.json`);
      platformSupport = await platformResponse.json();
      console.log('✅ Loaded platform support data');
    } catch (error) {
      console.warn('⚠️  platform-support.json not found, all models will be marked as experimental');
    }

    // 4. Fetch and parse each TOML file
    const models = [];
    for (const file of tomlFiles) {
      const modelId = file.name.replace('.toml', '');
      console.log(`  Processing ${modelId}...`);

      try {
        // Fetch raw TOML content
        const tomlResponse = await fetchData(`${GITHUB_RAW_URL}/${file.name}`);
        const tomlContent = await tomlResponse.text();

        // Parse TOML
        const tomlData = toml.parse(tomlContent);

        // Extract metadata
        const paramSize = extractParameterSize(modelId);
        const isVision = isVisionModel(tomlData);
        const huggingfaceUrl = extractHuggingFaceUrl(tomlData);
        const tags = extractTags(modelId, tomlData, isVision);

        // Get platform support (default to empty array if not specified)
        const platforms = platformSupport[modelId] || [];
        const isExperimental = platforms.length === 0;

        models.push({
          id: modelId,
          name: tomlData.name || modelId,
          description: tomlData.description || '',
          version: tomlData.version || '1.0',
          parameters: paramSize,
          platforms: platforms,
          isExperimental: isExperimental,
          tags: tags,
          isVision: isVision,
          huggingfaceUrl: huggingfaceUrl,
        });

      } catch (error) {
        console.error(`  ❌ Failed to process ${modelId}:`, error.message);
      }
    }

    // 5. Sort models by name
    models.sort((a, b) => a.name.localeCompare(b.name));

    // 6. Write to output file
    const outputDir = path.dirname(OUTPUT_FILE);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(models, null, 2));
    console.log(`✅ Generated ${OUTPUT_FILE} with ${models.length} models`);

    // Print summary
    const experimentalCount = models.filter(m => m.isExperimental).length;
    const visionCount = models.filter(m => m.isVision).length;
    console.log(`
📊 Summary:
   Total models: ${models.length}
   Vision models: ${visionCount}
   Experimental: ${experimentalCount}
   With platform support: ${models.length - experimentalCount}
`);

  } catch (error) {
    console.error('❌ Error fetching models:', error);
    process.exit(1);
  }
}

// Run the script
fetchModels();
