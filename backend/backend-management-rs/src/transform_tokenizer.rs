//! Tokenizer transformation utilities
//!
//! This module provides functionality to convert Hugging Face tokenizer configurations
//! into Symphony's tokenizer.model format for BPE tokenizers.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokio::fs;
use tracing::{info, warn};
use crate::error::{ManagementError, Result};

/// Hugging Face tokenizer JSON structure (partial)
#[derive(Debug, Deserialize)]
struct HfTokenizerJson {
    model: HfTokenizerModel,
    #[serde(default)]
    added_tokens: Vec<HfAddedToken>,
}

#[derive(Debug, Deserialize)]
struct HfTokenizerModel {
    #[serde(rename = "type")]
    model_type: String,
    vocab: HashMap<String, u32>,
    merges: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct HfAddedToken {
    id: u32,
    content: String,
    special: Option<bool>,
}

/// Special tokens configuration for Symphony
#[derive(Debug, Serialize, Deserialize)]
pub struct SpecialTokensConfig {
    pub tokens: HashMap<String, u32>,
}

/// Convert HuggingFace tokenizer to Symphony tokenizer.model format
pub async fn convert_hf_tokenizer_to_symphony(model_path: &Path) -> Result<()> {
    info!("Converting HF tokenizer to Symphony format in: {:?}", model_path);

    let tokenizer_json_path = model_path.join("tokenizer.json");
    if !tokenizer_json_path.exists() {
        return Err(ManagementError::Service {
            message: "tokenizer.json not found in model directory".to_string(),
        });
    }

    // Parse HF tokenizer JSON
    let tokenizer_content = fs::read_to_string(&tokenizer_json_path).await
        .map_err(|e| ManagementError::Service {
            message: format!("Failed to read tokenizer.json: {}", e),
        })?;

    let hf_tokenizer: HfTokenizerJson = serde_json::from_str(&tokenizer_content)
        .map_err(|e| ManagementError::Service {
            message: format!("Failed to parse tokenizer.json: {}", e),
        })?;

    // Simply convert vocab from json into byte sequence
    let tokenizer_model_path = model_path.join("tokenizer.model");
    // Check if tokenizer.model already exists
    if tokenizer_model_path.exists() {
        return Err(ManagementError::Service {
            message: "tokenizer.model already exists, please remove it first".to_string(),
        });
    }
    convert_vocab_to_bytes(&hf_tokenizer.model.vocab, &tokenizer_model_path).await?;

    info!("Successfully converted HF tokenizer to Symphony format");
    Ok(())
}

/// Create inverse mapping from unicode back to original bytes
/// This reverses the OpenAI bytes_to_unicode() transformation
fn unicode_to_bytes() -> HashMap<char, u8> {
    let mut byte_to_unicode = HashMap::new();
    
    // The 188 integers that render fine in their original form and need no shifting
    let mut bs = Vec::new();
    bs.extend(b'!'..=b'~');  // 33..126
    bs.extend(0xa1..=0xac);  // 161..172 (¡..¬)
    bs.extend(0xae..=0xff);  // 174..255 (®..ÿ)
    
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    
    // Add the other 68 integers that need shifting
    let mut n = 0u32;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    
    // Create mapping from shifted unicode back to original bytes
    for (byte_val, unicode_val) in bs.iter().zip(cs.iter()) {
        let unicode_char = char::from_u32(*unicode_val).unwrap();
        byte_to_unicode.insert(unicode_char, *byte_val);
    }
    
    byte_to_unicode
}

/// Convert vocabulary from UTF-8 strings to bytes and serialize to tokenizer.model
async fn convert_vocab_to_bytes(vocab: &HashMap<String, u32>, output_path: &Path) -> Result<()> {
    info!("Converting vocabulary from UTF-8 to bytes format");
    
    let unicode_to_byte_map = unicode_to_bytes();
    let mut converted_vocab: HashMap<Vec<u8>, u32> = HashMap::new();
    
    for (token_str, token_id) in vocab {
        // Convert the UTF-8 token string back to its original byte representation
        let mut byte_sequence = Vec::new();
        
        for ch in token_str.chars() {
            if let Some(&byte_val) = unicode_to_byte_map.get(&ch) {
                byte_sequence.push(byte_val);
            } else {
                warn!("Character '{}' not found in unicode_to_bytes mapping, using UTF-8 encoding", ch);
                // Fallback: encode as UTF-8 bytes
                byte_sequence.extend(ch.to_string().as_bytes());
            }
        }
        
        converted_vocab.insert(byte_sequence, *token_id);
    }
    
    info!("Converted {} vocabulary entries from UTF-8 to bytes", converted_vocab.len());
    
    // Serialize the converted vocabulary to tokenizer.model
    let serialized_data = serialize_vocab_to_model(&converted_vocab)?;
    
    fs::write(output_path, serialized_data).await
        .map_err(|e| ManagementError::Service {
            message: format!("Failed to write tokenizer.model: {}", e),
        })?;
    
    info!("Successfully wrote tokenizer.model to: {:?}", output_path);
    Ok(())
}

/// Serialize the byte-based vocabulary to Symphony's tokenizer.model format
/// Format: base64_encoded_token_bytes token_id (one per line, sorted by token_id)
fn serialize_vocab_to_model(vocab: &HashMap<Vec<u8>, u32>) -> Result<Vec<u8>> {
    use base64::{Engine as _, engine::general_purpose};
    
    // Sort entries by token_id for consistent output
    let mut sorted_entries: Vec<_> = vocab.iter().collect();
    sorted_entries.sort_by_key(|(_, &id)| id);
    
    let mut lines = Vec::new();
    
    // Write each vocabulary entry as: base64_token token_id
    for (token_bytes, &token_id) in sorted_entries {
        let base64_token = general_purpose::STANDARD.encode(token_bytes);
        let line = format!("{} {}\n", base64_token, token_id);
        lines.push(line);
    }
    
    let output = lines.join("");
    Ok(output.into_bytes())
}
