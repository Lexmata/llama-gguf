//! Tokenizer implementations for text encoding/decoding
//!
//! This module provides tokenizer implementations loaded from GGUF metadata.
//! Supports BPE (Byte Pair Encoding) and SentencePiece tokenizers.

use std::collections::HashMap;

use crate::gguf::{GgufFile, MetadataValue};

/// Tokenizer type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerType {
    /// Byte Pair Encoding
    BPE,
    /// SentencePiece (Unigram)
    SentencePiece,
    /// WordPiece
    WordPiece,
    /// Unknown type
    Unknown,
}

impl TokenizerType {
    /// Parse tokenizer type from GGUF metadata
    pub fn from_gguf_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "llama" | "bpe" => Self::BPE,
            "gpt2" => Self::BPE,
            "sentencepiece" | "spm" => Self::SentencePiece,
            "wordpiece" | "bert" => Self::WordPiece,
            _ => Self::Unknown,
        }
    }
}

/// Token type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenType {
    /// Normal token
    #[default]
    Normal,
    /// Control token (special)
    Control,
    /// Byte fallback token
    Byte,
    /// Unknown token
    Unknown,
}

/// Special token IDs
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// Beginning of sequence token
    pub bos_token_id: u32,
    /// End of sequence token
    pub eos_token_id: u32,
    /// Padding token (optional)
    pub pad_token_id: Option<u32>,
    /// Unknown token (optional)
    pub unk_token_id: Option<u32>,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_token_id: 1,
            eos_token_id: 2,
            pad_token_id: None,
            unk_token_id: Some(0),
        }
    }
}

/// Tokenizer error
#[derive(thiserror::Error, Debug)]
pub enum TokenizerError {
    #[error("Missing tokenizer data in GGUF: {0}")]
    MissingData(String),

    #[error("Invalid token: {0}")]
    InvalidToken(String),

    #[error("Encoding error: {0}")]
    EncodingError(String),
}

pub type TokenizerResult<T> = Result<T, TokenizerError>;

/// Tokenizer loaded from GGUF metadata
#[derive(Debug)]
pub struct Tokenizer {
    /// Token vocabulary (token string -> token id)
    token_to_id: HashMap<String, u32>,
    /// Reverse vocabulary (token id -> token string)
    id_to_token: Vec<String>,
    /// Token scores for SentencePiece
    scores: Vec<f32>,
    /// Merge pairs for BPE with priority (lower = merge first)
    /// Maps (token1_id, token2_id) -> (merged_token_id, priority)
    merges: HashMap<(u32, u32), (u32, usize)>,
    /// Special tokens
    pub special_tokens: SpecialTokens,
    /// Tokenizer type
    pub tokenizer_type: TokenizerType,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Token types (for distinguishing normal, control, byte tokens)
    token_types: Vec<TokenType>,
}

impl Tokenizer {
    /// Load tokenizer from GGUF file
    pub fn from_gguf(gguf: &GgufFile) -> TokenizerResult<Self> {
        // Get tokenizer type
        let tokenizer_type = gguf
            .data
            .get_string("tokenizer.ggml.model")
            .map(TokenizerType::from_gguf_str)
            .unwrap_or(TokenizerType::BPE);

        // Load vocabulary
        let tokens = Self::load_tokens(gguf)?;
        let vocab_size = tokens.len();

        // Build token mappings
        let mut token_to_id = HashMap::with_capacity(vocab_size);
        let mut id_to_token = Vec::with_capacity(vocab_size);

        for (id, token) in tokens.into_iter().enumerate() {
            token_to_id.insert(token.clone(), id as u32);
            id_to_token.push(token);
        }

        // Load scores if available
        let scores = Self::load_scores(gguf, vocab_size);

        // Load token types
        let token_types = Self::load_token_types(gguf, vocab_size);

        // Load merges for BPE
        let merges = Self::load_merges(gguf, &token_to_id);

        // Load special tokens
        let special_tokens = Self::load_special_tokens(gguf);

        Ok(Self {
            token_to_id,
            id_to_token,
            scores,
            merges,
            special_tokens,
            tokenizer_type,
            vocab_size,
            token_types,
        })
    }

    /// Load tokens from GGUF
    fn load_tokens(gguf: &GgufFile) -> TokenizerResult<Vec<String>> {
        let tokens_value = gguf
            .data
            .metadata
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| TokenizerError::MissingData("tokenizer.ggml.tokens".into()))?;

        match tokens_value {
            MetadataValue::Array(arr) => {
                let mut tokens = Vec::with_capacity(arr.values.len());
                for value in &arr.values {
                    match value {
                        MetadataValue::String(s) => tokens.push(s.clone()),
                        _ => {
                            return Err(TokenizerError::MissingData(
                                "Expected string tokens".into(),
                            ))
                        }
                    }
                }
                Ok(tokens)
            }
            _ => Err(TokenizerError::MissingData(
                "Expected token array".into(),
            )),
        }
    }

    /// Load token scores from GGUF
    fn load_scores(gguf: &GgufFile, vocab_size: usize) -> Vec<f32> {
        let scores_value = gguf.data.metadata.get("tokenizer.ggml.scores");

        match scores_value {
            Some(MetadataValue::Array(arr)) => {
                let mut scores = Vec::with_capacity(arr.values.len());
                for value in &arr.values {
                    match value {
                        MetadataValue::Float32(f) => scores.push(*f),
                        _ => scores.push(0.0),
                    }
                }
                scores
            }
            _ => vec![0.0; vocab_size],
        }
    }

    /// Load token types from GGUF
    fn load_token_types(gguf: &GgufFile, vocab_size: usize) -> Vec<TokenType> {
        let types_value = gguf.data.metadata.get("tokenizer.ggml.token_type");

        match types_value {
            Some(MetadataValue::Array(arr)) => {
                let mut types = Vec::with_capacity(arr.values.len());
                for value in &arr.values {
                    let token_type = match value {
                        MetadataValue::Int32(t) => match *t {
                            1 => TokenType::Normal,
                            2 => TokenType::Unknown,
                            3 => TokenType::Control,
                            6 => TokenType::Byte,
                            _ => TokenType::Normal,
                        },
                        _ => TokenType::Normal,
                    };
                    types.push(token_type);
                }
                types
            }
            _ => vec![TokenType::Normal; vocab_size],
        }
    }

    /// Load BPE merges from GGUF with priority ordering
    fn load_merges(
        gguf: &GgufFile,
        token_to_id: &HashMap<String, u32>,
    ) -> HashMap<(u32, u32), (u32, usize)> {
        let mut merges = HashMap::new();

        let merges_value = gguf.data.metadata.get("tokenizer.ggml.merges");

        if let Some(MetadataValue::Array(arr)) = merges_value {
            for (priority, value) in arr.values.iter().enumerate() {
                if let MetadataValue::String(merge_str) = value {
                    // Parse merge: "token1 token2"
                    let parts: Vec<&str> = merge_str.split(' ').collect();
                    if parts.len() == 2
                        && let (Some(&id1), Some(&id2)) =
                            (token_to_id.get(parts[0]), token_to_id.get(parts[1]))
                        {
                            // The merged result is typically the concatenation
                            let merged = format!("{}{}", parts[0], parts[1]);
                            if let Some(&merged_id) = token_to_id.get(&merged) {
                                merges.insert((id1, id2), (merged_id, priority));
                            }
                        }
                }
            }
        }

        merges
    }

    /// Load special tokens from GGUF
    fn load_special_tokens(gguf: &GgufFile) -> SpecialTokens {
        SpecialTokens {
            bos_token_id: gguf.data.get_u32("tokenizer.ggml.bos_token_id").unwrap_or(1),
            eos_token_id: gguf.data.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(2),
            pad_token_id: gguf.data.get_u32("tokenizer.ggml.padding_token_id"),
            unk_token_id: gguf.data.get_u32("tokenizer.ggml.unknown_token_id"),
        }
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_bos: bool) -> TokenizerResult<Vec<u32>> {
        let mut tokens = Vec::new();

        if add_bos {
            tokens.push(self.special_tokens.bos_token_id);
        }

        // Use BPE encoding if merges are available, otherwise use SentencePiece greedy
        if !self.merges.is_empty() {
            tokens.extend(self.encode_bpe(text)?);
        } else {
            // SentencePiece uses greedy longest-match algorithm
            tokens.extend(self.encode_sentencepiece(text)?);
        }

        Ok(tokens)
    }

    /// SentencePiece encoding using greedy longest-match algorithm
    fn encode_sentencepiece(&self, text: &str) -> TokenizerResult<Vec<u32>> {
        let mut result = Vec::new();
        
        // Add space prefix for LLaMA-style tokenizers
        let text_with_prefix = format!(" {}", text);
        let chars: Vec<char> = text_with_prefix.chars().collect();
        let mut pos = 0;
        
        while pos < chars.len() {
            let mut best_len = 0;
            let mut best_id = None;
            
            // Try to find the longest matching token starting at current position
            // Try lengths from longest to shortest for efficiency
            for end in (pos + 1..=chars.len()).rev() {
                let substr: String = chars[pos..end].iter().collect();
                
                // Try with SentencePiece space marker
                let spm_str = substr.replace(' ', "▁");
                if let Some(&id) = self.token_to_id.get(&spm_str) {
                    best_len = end - pos;
                    best_id = Some(id);
                    break; // Found longest match
                }
                
                // Try original string
                if let Some(&id) = self.token_to_id.get(&substr) {
                    best_len = end - pos;
                    best_id = Some(id);
                    break; // Found longest match
                }
            }
            
            if let Some(id) = best_id {
                result.push(id);
                pos += best_len;
            } else {
                // Fallback: try single character with byte fallback
                let ch = chars[pos];
                let ch_str = ch.to_string();
                
                // Try as SentencePiece space
                if ch == ' ' {
                    if let Some(&id) = self.token_to_id.get("▁") {
                        result.push(id);
                        pos += 1;
                        continue;
                    }
                }
                
                // Try as regular character
                if let Some(&id) = self.token_to_id.get(&ch_str) {
                    result.push(id);
                    pos += 1;
                    continue;
                }
                
                // Byte-level fallback
                for byte in ch_str.as_bytes() {
                    let byte_token = format!("<0x{:02X}>", byte);
                    if let Some(&id) = self.token_to_id.get(&byte_token) {
                        result.push(id);
                    } else if let Some(unk_id) = self.special_tokens.unk_token_id {
                        result.push(unk_id);
                    }
                }
                pos += 1;
            }
        }
        
        Ok(result)
    }

    /// BPE encoding algorithm
    fn encode_bpe(&self, text: &str) -> TokenizerResult<Vec<u32>> {
        // For LLaMA-style tokenizers, we first try to match whole words/subwords
        // then fall back to character-level
        
        let mut result = Vec::new();

        // Split text preserving whitespace (add space prefix for LLaMA)
        let text_with_prefix = if !text.starts_with(' ') && !text.is_empty() {
            format!(" {}", text)
        } else {
            text.to_string()
        };

        // Process each segment
        for segment in self.split_into_segments(&text_with_prefix) {
            if segment.is_empty() {
                continue;
            }

            // Try to find the segment directly in vocabulary
            if let Some(&id) = self.token_to_id.get(&segment) {
                result.push(id);
                continue;
            }

            // Convert to initial token sequence (characters or bytes)
            let mut tokens = self.text_to_initial_tokens(&segment)?;

            // Apply BPE merges iteratively
            loop {
                if tokens.len() < 2 {
                    break;
                }

                // Find the best merge (lowest priority number = highest priority)
                let mut best_merge: Option<(usize, u32, usize)> = None; // (position, merged_id, priority)

                for i in 0..tokens.len() - 1 {
                    let pair = (tokens[i], tokens[i + 1]);
                    if let Some(&(merged_id, priority)) = self.merges.get(&pair)
                        && (best_merge.is_none() || priority < best_merge.unwrap().2) {
                            best_merge = Some((i, merged_id, priority));
                        }
                }

                // Apply the best merge if found
                match best_merge {
                    Some((pos, merged_id, _)) => {
                        tokens[pos] = merged_id;
                        tokens.remove(pos + 1);
                    }
                    None => break, // No more merges possible
                }
            }

            result.extend(tokens);
        }

        Ok(result)
    }

    /// Split text into segments for BPE processing
    fn split_into_segments(&self, text: &str) -> Vec<String> {
        // Simple split: try to match known vocabulary entries greedily
        // For proper implementation, this should handle regex patterns per tokenizer type
        let mut segments = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            current.push(ch);
            
            // Check if current string exists in vocab or if it's a word boundary
            if (ch.is_whitespace() || ch.is_ascii_punctuation())
                && !current.is_empty() {
                    segments.push(current.clone());
                    current.clear();
                }
        }

        if !current.is_empty() {
            segments.push(current);
        }

        segments
    }

    /// Convert text segment to initial token sequence
    fn text_to_initial_tokens(&self, text: &str) -> TokenizerResult<Vec<u32>> {
        let mut tokens = Vec::new();

        for ch in text.chars() {
            let ch_str = ch.to_string();
            
            // First try direct character lookup
            if let Some(&id) = self.token_to_id.get(&ch_str) {
                tokens.push(id);
                continue;
            }

            // Try SentencePiece format (▁ prefix for space)
            if ch == ' '
                && let Some(&id) = self.token_to_id.get("▁") {
                    tokens.push(id);
                    continue;
                }

            // Try byte fallback
            for byte in ch_str.as_bytes() {
                let byte_token = format!("<0x{:02X}>", byte);
                if let Some(&id) = self.token_to_id.get(&byte_token) {
                    tokens.push(id);
                } else if let Some(unk_id) = self.special_tokens.unk_token_id {
                    tokens.push(unk_id);
                }
            }
        }

        Ok(tokens)
    }

    /// Fallback encoding (character/byte level) - kept for potential future use
    #[allow(dead_code)]
    fn encode_fallback(&self, text: &str) -> TokenizerResult<Vec<u32>> {
        let mut tokens = Vec::new();

        for ch in text.chars() {
            let ch_str = ch.to_string();
            if let Some(&id) = self.token_to_id.get(&ch_str) {
                tokens.push(id);
            } else {
                // Try byte fallback
                for byte in ch_str.as_bytes() {
                    let byte_token = format!("<0x{:02X}>", byte);
                    if let Some(&id) = self.token_to_id.get(&byte_token) {
                        tokens.push(id);
                    } else if let Some(unk_id) = self.special_tokens.unk_token_id {
                        tokens.push(unk_id);
                    }
                }
            }
        }

        Ok(tokens)
    }

    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[u32]) -> TokenizerResult<String> {
        let mut text = String::new();
        let mut byte_buffer: Vec<u8> = Vec::new();

        for &token_id in tokens {
            // Skip special tokens in output
            if token_id == self.special_tokens.bos_token_id
                || token_id == self.special_tokens.eos_token_id
            {
                continue;
            }

            if let Some(pad_id) = self.special_tokens.pad_token_id
                && token_id == pad_id {
                    continue;
                }

            let token_str = self
                .id_to_token
                .get(token_id as usize)
                .ok_or_else(|| TokenizerError::InvalidToken(format!("Unknown token ID: {}", token_id)))?;

            // Handle byte tokens - collect into buffer for proper UTF-8 decoding
            if token_str.starts_with("<0x") && token_str.ends_with('>') && token_str.len() == 6
                && let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16) {
                    byte_buffer.push(byte);
                    continue;
                }

            // Flush byte buffer before adding text
            if !byte_buffer.is_empty() {
                if let Ok(s) = String::from_utf8(byte_buffer.clone()) {
                    text.push_str(&s);
                } else {
                    // Invalid UTF-8, decode as lossy
                    text.push_str(&String::from_utf8_lossy(&byte_buffer));
                }
                byte_buffer.clear();
            }

            // Handle BPE special characters
            let decoded = token_str
                .replace("▁", " ")   // SentencePiece space
                .replace("Ġ", " ")   // GPT-2 style space  
                .replace("Ċ", "\n")  // GPT-2 style newline
                .replace("ĉ", "\t"); // GPT-2 style tab

            // Debug: uncomment to see token decoding
            // eprintln!("Token {}: '{}' -> '{}' bytes:{:?}", token_id, token_str, decoded, token_str.as_bytes());
            
            text.push_str(&decoded);
        }

        // Flush remaining bytes
        if !byte_buffer.is_empty() {
            if let Ok(s) = String::from_utf8(byte_buffer.clone()) {
                text.push_str(&s);
            } else {
                text.push_str(&String::from_utf8_lossy(&byte_buffer));
            }
        }

        Ok(text)
    }

    /// Decode a single token to string
    pub fn decode_token(&self, token_id: u32) -> TokenizerResult<String> {
        self.decode(&[token_id])
    }

    /// Get token string by ID
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    /// Get token ID by string
    pub fn get_token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token type
    pub fn get_token_type(&self, id: u32) -> TokenType {
        self.token_types
            .get(id as usize)
            .copied()
            .unwrap_or(TokenType::Normal)
    }

    /// Check if a token is a special token
    pub fn is_special_token(&self, id: u32) -> bool {
        id == self.special_tokens.bos_token_id
            || id == self.special_tokens.eos_token_id
            || self.special_tokens.pad_token_id == Some(id)
            || self.special_tokens.unk_token_id == Some(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_type_parsing() {
        assert_eq!(TokenizerType::from_gguf_str("llama"), TokenizerType::BPE);
        assert_eq!(TokenizerType::from_gguf_str("bpe"), TokenizerType::BPE);
        assert_eq!(
            TokenizerType::from_gguf_str("sentencepiece"),
            TokenizerType::SentencePiece
        );
    }

    #[test]
    fn test_special_tokens_default() {
        let special = SpecialTokens::default();
        assert_eq!(special.bos_token_id, 1);
        assert_eq!(special.eos_token_id, 2);
    }
}
