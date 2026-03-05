//! Tokenizer implementations for text encoding/decoding
//!
//! This module provides tokenizer implementations loaded from GGUF metadata.
//! Supports BPE (Byte Pair Encoding) and SentencePiece tokenizers.

use std::collections::HashMap;

use unicode_normalization::UnicodeNormalization;

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

/// Text normalizer applied before tokenization (from HuggingFace tokenizer.json)
#[derive(Debug, Clone)]
pub enum Normalizer {
    NFC,
    NFKC,
    NFD,
    NFKD,
    Lowercase,
    Strip { left: bool, right: bool },
    Prepend(String),
    Replace { pattern: String, content: String },
    StripAccents,
    Sequence(Vec<Normalizer>),
}

impl Normalizer {
    fn apply(&self, text: &str) -> String {
        match self {
            Self::NFC => text.nfc().collect(),
            Self::NFKC => text.nfkc().collect(),
            Self::NFD => text.nfd().collect(),
            Self::NFKD => text.nfkd().collect(),
            Self::Lowercase => text.to_lowercase(),
            Self::Strip { left, right } => {
                let s = if *left { text.trim_start() } else { text };
                if *right { s.trim_end().to_string() } else { s.to_string() }
            }
            Self::Prepend(prefix) => format!("{}{}", prefix, text),
            Self::Replace { pattern, content } => text.replace(pattern.as_str(), content.as_str()),
            Self::StripAccents => {
                text.nfkd()
                    .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
                    .collect()
            }
            Self::Sequence(normalizers) => {
                let mut result = text.to_string();
                for n in normalizers {
                    result = n.apply(&result);
                }
                result
            }
        }
    }
}

/// Pre-tokenizer splits text into segments before model encoding
#[derive(Debug, Clone)]
pub enum PreTokenizer {
    /// GPT-2 style byte-level splitting
    ByteLevel { add_prefix_space: bool },
    /// Split on whitespace boundaries
    Whitespace,
    /// SentencePiece metaspace handling
    Metaspace { replacement: char, add_prefix_space: bool },
    /// Split around punctuation characters
    Punctuation,
    /// Split digits
    Digits { individual_digits: bool },
    /// Chain multiple pre-tokenizers
    Sequence(Vec<PreTokenizer>),
}

impl PreTokenizer {
    fn apply(&self, text: &str) -> Vec<String> {
        match self {
            Self::ByteLevel { add_prefix_space } => {
                let text = if *add_prefix_space && !text.starts_with(' ') {
                    format!(" {}", text)
                } else {
                    text.to_string()
                };
                let mut tokens = Vec::new();
                let mut current = String::new();
                for ch in text.chars() {
                    if ch == ' ' && !current.is_empty() {
                        tokens.push(std::mem::take(&mut current));
                    }
                    current.push(ch);
                }
                if !current.is_empty() {
                    tokens.push(current);
                }
                tokens
            }
            Self::Whitespace => {
                text.split_whitespace().map(|s| s.to_string()).collect()
            }
            Self::Metaspace { replacement, add_prefix_space } => {
                let text = if *add_prefix_space && !text.starts_with(' ') {
                    format!(" {}", text)
                } else {
                    text.to_string()
                };
                text.split(' ')
                    .enumerate()
                    .filter(|(_, s)| !s.is_empty() || true)
                    .map(|(i, s)| {
                        if i == 0 && s.is_empty() {
                            replacement.to_string()
                        } else if i > 0 {
                            format!("{}{}", replacement, s)
                        } else {
                            s.to_string()
                        }
                    })
                    .filter(|s| !s.is_empty())
                    .collect()
            }
            Self::Punctuation => {
                let mut result = Vec::new();
                let mut current = String::new();
                for ch in text.chars() {
                    if ch.is_ascii_punctuation() {
                        if !current.is_empty() {
                            result.push(std::mem::take(&mut current));
                        }
                        result.push(ch.to_string());
                    } else {
                        current.push(ch);
                    }
                }
                if !current.is_empty() {
                    result.push(current);
                }
                result
            }
            Self::Digits { individual_digits } => {
                if !*individual_digits {
                    return vec![text.to_string()];
                }
                let mut result = Vec::new();
                let mut current = String::new();
                for ch in text.chars() {
                    if ch.is_ascii_digit() {
                        if !current.is_empty() {
                            result.push(std::mem::take(&mut current));
                        }
                        result.push(ch.to_string());
                    } else {
                        current.push(ch);
                    }
                }
                if !current.is_empty() {
                    result.push(current);
                }
                result
            }
            Self::Sequence(pre_tokenizers) => {
                let mut segments = vec![text.to_string()];
                for pt in pre_tokenizers {
                    let mut next = Vec::new();
                    for seg in &segments {
                        next.extend(pt.apply(seg));
                    }
                    segments = next;
                }
                segments
            }
        }
    }
}

/// Element in a template processing sequence
#[derive(Debug, Clone)]
pub enum TemplateElement {
    SpecialToken { id: String, token_id: u32 },
    Sequence { type_id: u32 },
}

/// Post-processor adds special tokens after encoding
#[derive(Debug, Clone)]
pub enum PostProcessor {
    TemplateProcessing {
        single: Vec<TemplateElement>,
        pair: Vec<TemplateElement>,
    },
    ByteLevel { trim_offsets: bool },
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

/// Extract the longest valid UTF-8 prefix from `buf`, draining those bytes.
/// Bytes that form incomplete trailing sequences are left in `buf` for the
/// next call. At most 3 trailing bytes can remain (start of a 2–4 byte seq).
fn flush_valid_utf8(buf: &mut Vec<u8>) -> String {
    if buf.is_empty() {
        return String::new();
    }

    // Find the longest prefix that is valid UTF-8.
    // An incomplete trailing multi-byte sequence has at most 3 leading bytes.
    let valid_up_to = match std::str::from_utf8(buf) {
        Ok(_) => {
            let s = String::from_utf8(std::mem::take(buf)).unwrap();
            return s;
        }
        Err(e) => e.valid_up_to(),
    };

    if valid_up_to == 0 {
        // Check if the entire buffer is a partial multi-byte start (≤3 bytes)
        // that could become valid with more bytes
        if buf.len() <= 3 && buf[0] >= 0x80 {
            return String::new();
        }
        // Otherwise, the first byte is truly invalid — emit replacement and skip it
        buf.remove(0);
        return String::from("\u{FFFD}");
    }

    let text = String::from_utf8(buf[..valid_up_to].to_vec()).unwrap();
    *buf = buf[valid_up_to..].to_vec();
    text
}

/// Build both directions of the GPT-2 byte ↔ unicode mapping.
///
/// GPT-2 BPE maps every byte (0-255) to a printable Unicode character so that
/// token strings are always valid Unicode. Printable ASCII and certain Latin-1
/// bytes map to themselves; the remaining 68 bytes map to U+0100..U+0143.
fn build_gpt2_mappings() -> (HashMap<char, u8>, [char; 256]) {
    let mut byte_to_unicode = ['\0'; 256];

    let mut direct: Vec<u8> = Vec::new();
    direct.extend(33u8..=126);
    direct.extend(161u8..=172);
    direct.extend(174u8..=255);

    for &b in &direct {
        byte_to_unicode[b as usize] = char::from(b);
    }

    let mut n: u32 = 0;
    for b in 0u16..=255 {
        if !direct.contains(&(b as u8)) {
            byte_to_unicode[b as usize] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }

    let unicode_to_byte: HashMap<char, u8> = byte_to_unicode
        .iter()
        .enumerate()
        .map(|(b, &c)| (c, b as u8))
        .collect();

    (unicode_to_byte, byte_to_unicode)
}

/// A segment of text that has been split around special/control tokens.
#[derive(Debug, Clone)]
enum TextSegment {
    /// Regular text to be encoded with BPE/SentencePiece
    Text(String),
    /// A control/special token that maps directly to a token ID
    SpecialToken(u32),
}

/// Tokenizer loaded from GGUF metadata or HuggingFace tokenizer.json
#[derive(Debug)]
pub struct Tokenizer {
    /// Token vocabulary (token string -> token id)
    token_to_id: HashMap<String, u32>,
    /// Reverse vocabulary (token id -> token string)
    id_to_token: Vec<String>,
    /// Token scores (log probabilities for Unigram models)
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
    /// GPT-2 unicode-to-byte reverse mapping (only for GPT-2 tokenizers)
    gpt2_unicode_to_byte: Option<HashMap<char, u8>>,
    /// GPT-2 byte-to-unicode forward mapping for encoding (only for GPT-2 tokenizers)
    gpt2_byte_to_unicode: Option<[char; 256]>,
    /// HF normalizer pipeline component
    normalizer: Option<Normalizer>,
    /// HF pre-tokenizer pipeline component
    pre_tokenizer: Option<PreTokenizer>,
    /// HF post-processor pipeline component
    post_processor: Option<PostProcessor>,
    /// WordPiece continuation prefix (default "##")
    wordpiece_prefix: String,
    /// Control token strings sorted by length (longest first) for greedy matching
    control_token_strings: Vec<(String, u32)>,
    /// Whether the GGUF explicitly defined a BOS token ID
    pub has_explicit_bos: bool,
}

impl Tokenizer {
    /// Load tokenizer from GGUF file
    pub fn from_gguf(gguf: &GgufFile) -> TokenizerResult<Self> {
        // Get tokenizer type
        let model_str = gguf
            .data
            .get_string("tokenizer.ggml.model")
            .unwrap_or("bpe");
        let tokenizer_type = TokenizerType::from_gguf_str(model_str);

        // GPT-2 style tokenizers use byte-level BPE with a unicode mapping
        let uses_gpt2_bytes = model_str == "gpt2"
            || gguf
                .data
                .get_string("tokenizer.ggml.pre")
                .is_some_and(|p| {
                    matches!(
                        p,
                        "qwen2" | "gpt-2" | "gpt2" | "starcoder" | "deepseek-llm" | "deepseek-coder"
                    )
                });

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

        let (gpt2_unicode_to_byte, gpt2_byte_to_unicode) = if uses_gpt2_bytes {
            let (u2b, b2u) = build_gpt2_mappings();
            (Some(u2b), Some(b2u))
        } else {
            (None, None)
        };

        let has_explicit_bos = gguf.data.get_u32("tokenizer.ggml.bos_token_id").is_some();

        let mut control_token_strings: Vec<(String, u32)> = token_types
            .iter()
            .enumerate()
            .filter(|(_, tt)| **tt == TokenType::Control)
            .filter_map(|(id, _)| {
                let s = &id_to_token[id];
                if !s.is_empty() {
                    Some((s.clone(), id as u32))
                } else {
                    None
                }
            })
            .collect();
        control_token_strings.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        Ok(Self {
            token_to_id,
            id_to_token,
            scores,
            merges,
            special_tokens,
            tokenizer_type,
            vocab_size,
            token_types,
            gpt2_unicode_to_byte,
            gpt2_byte_to_unicode,
            normalizer: None,
            pre_tokenizer: None,
            post_processor: None,
            wordpiece_prefix: "##".to_string(),
            control_token_strings,
            has_explicit_bos,
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
                            ));
                        }
                    }
                }
                Ok(tokens)
            }
            _ => Err(TokenizerError::MissingData("Expected token array".into())),
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
            bos_token_id: gguf
                .data
                .get_u32("tokenizer.ggml.bos_token_id")
                .unwrap_or(1),
            eos_token_id: gguf
                .data
                .get_u32("tokenizer.ggml.eos_token_id")
                .unwrap_or(2),
            pad_token_id: gguf.data.get_u32("tokenizer.ggml.padding_token_id"),
            unk_token_id: gguf.data.get_u32("tokenizer.ggml.unknown_token_id"),
        }
    }

    /// Split text around control/special token strings.
    ///
    /// Scans `text` for any control token literal (e.g. `<|im_start|>`) and
    /// splits it into alternating Text / SpecialToken segments. Uses greedy
    /// longest-match so longer control tokens take priority.
    fn split_with_special_tokens(&self, text: &str) -> Vec<TextSegment> {
        if self.control_token_strings.is_empty() {
            return vec![TextSegment::Text(text.to_string())];
        }

        let mut segments = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            let mut earliest_pos = remaining.len();
            let mut matched_len = 0;
            let mut matched_id = 0u32;

            for (tok_str, tok_id) in &self.control_token_strings {
                if let Some(pos) = remaining.find(tok_str.as_str()) {
                    if pos < earliest_pos
                        || (pos == earliest_pos && tok_str.len() > matched_len)
                    {
                        earliest_pos = pos;
                        matched_len = tok_str.len();
                        matched_id = *tok_id;
                    }
                }
            }

            if matched_len == 0 {
                segments.push(TextSegment::Text(remaining.to_string()));
                break;
            }

            if earliest_pos > 0 {
                segments.push(TextSegment::Text(remaining[..earliest_pos].to_string()));
            }
            segments.push(TextSegment::SpecialToken(matched_id));
            remaining = &remaining[earliest_pos + matched_len..];
        }

        segments
    }

    /// Encode a plain text segment (no special tokens) using the appropriate algorithm.
    fn encode_text_segment(&self, text: &str) -> TokenizerResult<Vec<u32>> {
        if text.is_empty() {
            return Ok(vec![]);
        }
        if self.normalizer.is_some() || self.pre_tokenizer.is_some() {
            let normalized = match &self.normalizer {
                Some(n) => n.apply(text),
                None => text.to_string(),
            };
            let pre_tokens = match &self.pre_tokenizer {
                Some(pt) => pt.apply(&normalized),
                None => vec![normalized],
            };
            let mut tokens = Vec::new();
            for pre_token in &pre_tokens {
                if pre_token.is_empty() {
                    continue;
                }
                match self.tokenizer_type {
                    TokenizerType::SentencePiece => {
                        tokens.extend(self.encode_unigram(pre_token)?);
                    }
                    TokenizerType::WordPiece => {
                        tokens.extend(self.encode_wordpiece(pre_token)?);
                    }
                    _ => {
                        tokens.extend(self.encode_bpe_pretokenized(pre_token)?);
                    }
                }
            }
            Ok(tokens)
        } else if !self.merges.is_empty() {
            self.encode_bpe(text)
        } else {
            self.encode_sentencepiece(text)
        }
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_bos: bool) -> TokenizerResult<Vec<u32>> {
        let mut tokens = Vec::new();

        if add_bos {
            tokens.push(self.special_tokens.bos_token_id);
        }

        let segments = self.split_with_special_tokens(text);
        for segment in segments {
            match segment {
                TextSegment::Text(t) => {
                    tokens.extend(self.encode_text_segment(&t)?);
                }
                TextSegment::SpecialToken(id) => {
                    tokens.push(id);
                }
            }
        }

        if !add_bos {
            if let Some(PostProcessor::TemplateProcessing { ref single, .. }) = self.post_processor {
                let mut processed = Vec::new();
                for elem in single {
                    match elem {
                        TemplateElement::SpecialToken { token_id, .. } => {
                            processed.push(*token_id);
                        }
                        TemplateElement::Sequence { .. } => {
                            processed.extend(&tokens);
                        }
                    }
                }
                return Ok(processed);
            }
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
                if ch == ' '
                    && let Some(&id) = self.token_to_id.get("▁")
                {
                    result.push(id);
                    pos += 1;
                    continue;
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
        if self.gpt2_byte_to_unicode.is_some() {
            return self.encode_bpe_gpt2(text);
        }

        let mut result = Vec::new();

        let text_with_prefix = if !text.starts_with(' ') && !text.is_empty() {
            format!(" {}", text)
        } else {
            text.to_string()
        };

        for segment in self.split_into_segments(&text_with_prefix) {
            if segment.is_empty() {
                continue;
            }

            if let Some(&id) = self.token_to_id.get(&segment) {
                result.push(id);
                continue;
            }

            let mut tokens = self.text_to_initial_tokens(&segment)?;
            self.apply_bpe_merges(&mut tokens);
            result.extend(tokens);
        }

        Ok(result)
    }

    /// GPT-2 byte-level BPE encoding.
    ///
    /// Converts input bytes through the GPT-2 byte→unicode mapping, splits
    /// into pretokenized segments, and applies BPE merges on each.
    fn encode_bpe_gpt2(&self, text: &str) -> TokenizerResult<Vec<u32>> {
        let b2u = self.gpt2_byte_to_unicode.as_ref().unwrap();
        let mut result = Vec::new();

        for segment in Self::gpt2_pretokenize(text) {
            if segment.is_empty() {
                continue;
            }

            let mapped: String = segment.as_bytes().iter().map(|&b| b2u[b as usize]).collect();

            if let Some(&id) = self.token_to_id.get(&mapped) {
                result.push(id);
                continue;
            }

            let mut tokens: Vec<u32> = Vec::with_capacity(mapped.len());
            for ch in mapped.chars() {
                let ch_str = ch.to_string();
                if let Some(&id) = self.token_to_id.get(&ch_str) {
                    tokens.push(id);
                } else if let Some(unk_id) = self.special_tokens.unk_token_id {
                    tokens.push(unk_id);
                }
            }

            self.apply_bpe_merges(&mut tokens);
            result.extend(tokens);
        }

        Ok(result)
    }

    /// Simple GPT-2 pretokenization: split text into chunks at word boundaries.
    ///
    /// Spaces attach to the following word. Newlines and other control
    /// characters are their own chunks. Runs of letters, runs of digits
    /// (up to 3), and individual punctuation are separate chunks.
    fn gpt2_pretokenize(text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let ch = chars[i];

            if ch == ' ' {
                let mut chunk = String::new();
                chunk.push(ch);
                i += 1;
                if i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    while i < chars.len()
                        && !chars[i].is_whitespace()
                        && (chars[i].is_alphanumeric() || chars[i] == '_')
                    {
                        chunk.push(chars[i]);
                        i += 1;
                    }
                }
                chunks.push(chunk);
            } else if ch == '\n' || ch == '\r' || ch == '\t' {
                let mut chunk = String::new();
                while i < chars.len()
                    && (chars[i] == '\n' || chars[i] == '\r' || chars[i] == '\t')
                {
                    chunk.push(chars[i]);
                    i += 1;
                }
                chunks.push(chunk);
            } else if ch.is_alphabetic() || ch == '_' {
                let mut chunk = String::new();
                while i < chars.len() && (chars[i].is_alphabetic() || chars[i] == '_') {
                    chunk.push(chars[i]);
                    i += 1;
                }
                chunks.push(chunk);
            } else if ch.is_ascii_digit() {
                let mut chunk = String::new();
                let mut count = 0;
                while i < chars.len() && chars[i].is_ascii_digit() && count < 3 {
                    chunk.push(chars[i]);
                    i += 1;
                    count += 1;
                }
                chunks.push(chunk);
            } else {
                chunks.push(ch.to_string());
                i += 1;
            }
        }

        chunks
    }

    /// Apply BPE merges iteratively until no more merges are possible.
    fn apply_bpe_merges(&self, tokens: &mut Vec<u32>) {
        loop {
            if tokens.len() < 2 {
                break;
            }

            let mut best_merge: Option<(usize, u32, usize)> = None;

            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i], tokens[i + 1]);
                if let Some(&(merged_id, priority)) = self.merges.get(&pair)
                    && (best_merge.is_none() || priority < best_merge.unwrap().2)
                {
                    best_merge = Some((i, merged_id, priority));
                }
            }

            match best_merge {
                Some((pos, merged_id, _)) => {
                    tokens[pos] = merged_id;
                    tokens.remove(pos + 1);
                }
                None => break,
            }
        }
    }

    /// Split text into segments for non-GPT-2 BPE processing
    fn split_into_segments(&self, text: &str) -> Vec<String> {
        let mut segments = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            current.push(ch);

            if (ch.is_whitespace() || ch.is_ascii_punctuation()) && !current.is_empty() {
                segments.push(current.clone());
                current.clear();
            }
        }

        if !current.is_empty() {
            segments.push(current);
        }

        segments
    }

    /// Convert text segment to initial token sequence (non-GPT-2 path)
    fn text_to_initial_tokens(&self, text: &str) -> TokenizerResult<Vec<u32>> {
        let mut tokens = Vec::new();

        for ch in text.chars() {
            let ch_str = ch.to_string();

            if let Some(&id) = self.token_to_id.get(&ch_str) {
                tokens.push(id);
                continue;
            }

            if ch == ' '
                && let Some(&id) = self.token_to_id.get("▁")
            {
                tokens.push(id);
                continue;
            }

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

    /// Unigram encoding using Viterbi algorithm (for HuggingFace Unigram models)
    fn encode_unigram(&self, text: &str) -> TokenizerResult<Vec<u32>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        let char_boundaries: Vec<usize> = text
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(text.len()))
            .collect();
        let n = char_boundaries.len() - 1;

        const NEG_INF: f64 = -1e18;
        let mut best_score = vec![NEG_INF; n + 1];
        let mut best_path: Vec<Option<(u32, usize)>> = vec![None; n + 1];
        best_score[0] = 0.0;

        let max_token_chars = 128;

        for end in 1..=n {
            let end_byte = char_boundaries[end];
            let min_start = end.saturating_sub(max_token_chars);

            for start in (min_start..end).rev() {
                if best_score[start] <= NEG_INF {
                    continue;
                }
                let start_byte = char_boundaries[start];
                let substr = &text[start_byte..end_byte];

                if let Some(&id) = self.token_to_id.get(substr) {
                    let score = *self.scores.get(id as usize).unwrap_or(&0.0) as f64;
                    let candidate = best_score[start] + score;
                    if candidate > best_score[end] {
                        best_score[end] = candidate;
                        best_path[end] = Some((id, start));
                    }
                }
            }

            // Single-char byte fallback if no token found
            if best_path[end].is_none() && best_score[end - 1] > NEG_INF {
                let start_byte = char_boundaries[end - 1];
                let end_byte_val = char_boundaries[end];
                let ch_str = &text[start_byte..end_byte_val];

                if let Some(&id) = self.token_to_id.get(ch_str) {
                    let score = *self.scores.get(id as usize).unwrap_or(&-10.0) as f64;
                    best_score[end] = best_score[end - 1] + score;
                    best_path[end] = Some((id, end - 1));
                } else {
                    // Try byte-level fallback tokens
                    for byte in ch_str.as_bytes() {
                        let byte_token = format!("<0x{:02X}>", byte);
                        if let Some(&id) = self.token_to_id.get(&byte_token) {
                            let score = *self.scores.get(id as usize).unwrap_or(&-10.0) as f64;
                            let candidate = best_score[end - 1] + score;
                            if candidate > best_score[end] {
                                best_score[end] = candidate;
                                best_path[end] = Some((id, end - 1));
                            }
                        }
                    }
                }
            }
        }

        if best_score[n] <= NEG_INF {
            return self.encode_unigram_fallback(text);
        }

        let mut result = Vec::new();
        let mut pos = n;
        while pos > 0 {
            if let Some((token_id, start)) = best_path[pos] {
                result.push(token_id);
                pos = start;
            } else {
                break;
            }
        }
        result.reverse();
        Ok(result)
    }

    /// Fallback for Unigram when Viterbi cannot find a complete path
    fn encode_unigram_fallback(&self, text: &str) -> TokenizerResult<Vec<u32>> {
        let mut result = Vec::new();
        for ch in text.chars() {
            let ch_str = ch.to_string();
            if let Some(&id) = self.token_to_id.get(&ch_str) {
                result.push(id);
            } else {
                for byte in ch_str.as_bytes() {
                    let byte_token = format!("<0x{:02X}>", byte);
                    if let Some(&id) = self.token_to_id.get(&byte_token) {
                        result.push(id);
                    } else if let Some(unk_id) = self.special_tokens.unk_token_id {
                        result.push(unk_id);
                    }
                }
            }
        }
        Ok(result)
    }

    /// WordPiece encoding using greedy longest-match with continuation prefix
    fn encode_wordpiece(&self, text: &str) -> TokenizerResult<Vec<u32>> {
        if text.is_empty() {
            return Ok(vec![]);
        }

        let mut result = Vec::new();
        let chars: Vec<char> = text.chars().collect();

        // WordPiece operates on whitespace-split words
        let words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
        let words = if words.is_empty() {
            vec![text.to_string()]
        } else {
            words
        };

        for word in &words {
            let word_chars: Vec<char> = word.chars().collect();
            if word_chars.len() > 200 {
                if let Some(unk_id) = self.special_tokens.unk_token_id {
                    result.push(unk_id);
                }
                continue;
            }

            let mut start = 0;
            let mut is_first_subword = true;

            while start < word_chars.len() {
                let mut end = word_chars.len();
                let mut found = false;

                while start < end {
                    let substr: String = word_chars[start..end].iter().collect();
                    let candidate = if is_first_subword {
                        substr.clone()
                    } else {
                        format!("{}{}", self.wordpiece_prefix, substr)
                    };

                    if let Some(&id) = self.token_to_id.get(&candidate) {
                        result.push(id);
                        found = true;
                        break;
                    }
                    end -= 1;
                }

                if !found {
                    if let Some(unk_id) = self.special_tokens.unk_token_id {
                        result.push(unk_id);
                    }
                    break;
                }

                start = end;
                is_first_subword = false;
            }
        }

        let _ = chars; // suppress unused warning from pre-existing binding
        Ok(result)
    }

    /// BPE encoding for a pre-tokenized segment (no further splitting)
    fn encode_bpe_pretokenized(&self, text: &str) -> TokenizerResult<Vec<u32>> {
        if let Some(&id) = self.token_to_id.get(text) {
            return Ok(vec![id]);
        }

        let mut tokens = self.text_to_initial_tokens(text)?;
        self.apply_bpe_merges(&mut tokens);
        Ok(tokens)
    }

    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[u32]) -> TokenizerResult<String> {
        if let Some(ref u2b) = self.gpt2_unicode_to_byte {
            return self.decode_gpt2(tokens, u2b);
        }
        self.decode_sentencepiece(tokens)
    }

    /// Decode for GPT-2 byte-level BPE tokenizers (Qwen, StarCoder, DeepSeek, etc.)
    ///
    /// Each character in a GPT-2 token string represents one byte via the byte_to_unicode
    /// mapping. Decoding reverses the mapping to recover the raw byte sequence, then
    /// interprets those bytes as UTF-8.
    fn decode_gpt2(
        &self,
        tokens: &[u32],
        unicode_to_byte: &HashMap<char, u8>,
    ) -> TokenizerResult<String> {
        let mut raw_bytes: Vec<u8> = Vec::new();

        for &token_id in tokens {
            if self.is_special_token(token_id) {
                continue;
            }

            let token_str = self.id_to_token.get(token_id as usize).ok_or_else(|| {
                TokenizerError::InvalidToken(format!("Unknown token ID: {}", token_id))
            })?;

            // Skip control tokens that render as literal text (e.g. <|im_end|>)
            if self.get_token_type(token_id) == TokenType::Control {
                continue;
            }

            // Handle <0x??> byte fallback tokens
            if token_str.starts_with("<0x")
                && token_str.ends_with('>')
                && token_str.len() == 6
                && let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16)
            {
                raw_bytes.push(byte);
                continue;
            }

            // Map each character through the GPT-2 unicode→byte table
            for ch in token_str.chars() {
                if let Some(&b) = unicode_to_byte.get(&ch) {
                    raw_bytes.push(b);
                } else {
                    // Character not in the GPT-2 table — encode its UTF-8 bytes directly
                    let mut buf = [0u8; 4];
                    let encoded = ch.encode_utf8(&mut buf);
                    raw_bytes.extend_from_slice(encoded.as_bytes());
                }
            }
        }

        Ok(String::from_utf8_lossy(&raw_bytes).into_owned())
    }

    /// Decode for SentencePiece-style tokenizers (LLaMA, etc.)
    fn decode_sentencepiece(&self, tokens: &[u32]) -> TokenizerResult<String> {
        let mut text = String::new();
        let mut byte_buffer: Vec<u8> = Vec::new();

        for &token_id in tokens {
            if self.is_special_token(token_id) {
                continue;
            }

            if self.get_token_type(token_id) == TokenType::Control {
                continue;
            }

            let token_str = self.id_to_token.get(token_id as usize).ok_or_else(|| {
                TokenizerError::InvalidToken(format!("Unknown token ID: {}", token_id))
            })?;

            // Handle byte tokens — collect into buffer for proper UTF-8 decoding
            if token_str.starts_with("<0x")
                && token_str.ends_with('>')
                && token_str.len() == 6
                && let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16)
            {
                byte_buffer.push(byte);
                continue;
            }

            // Flush byte buffer before adding text
            if !byte_buffer.is_empty() {
                text.push_str(&String::from_utf8_lossy(&byte_buffer));
                byte_buffer.clear();
            }

            // SentencePiece uses ▁ for leading spaces
            text.push_str(&token_str.replace('▁', " "));
        }

        // Flush remaining bytes
        if !byte_buffer.is_empty() {
            text.push_str(&String::from_utf8_lossy(&byte_buffer));
        }

        Ok(text)
    }

    /// Decode a single token to string
    pub fn decode_token(&self, token_id: u32) -> TokenizerResult<String> {
        self.decode(&[token_id])
    }

    /// Decode a single token in streaming mode, handling incomplete UTF-8 sequences.
    ///
    /// For GPT-2 byte-level tokenizers, multi-byte UTF-8 characters (like emoji)
    /// may be split across multiple tokens. This method accumulates raw bytes in
    /// `pending` and only returns text once complete UTF-8 code points are formed.
    pub fn decode_token_streaming(
        &self,
        token_id: u32,
        pending: &mut Vec<u8>,
    ) -> TokenizerResult<String> {
        if self.is_special_token(token_id) || self.get_token_type(token_id) == TokenType::Control {
            // Flush any pending bytes before emitting nothing
            let flushed = flush_valid_utf8(pending);
            return Ok(flushed);
        }

        let token_str = self.id_to_token.get(token_id as usize).ok_or_else(|| {
            TokenizerError::InvalidToken(format!("Unknown token ID: {}", token_id))
        })?;

        // Handle <0x??> byte fallback tokens
        if token_str.starts_with("<0x")
            && token_str.ends_with('>')
            && token_str.len() == 6
            && let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16)
        {
            pending.push(byte);
            return Ok(flush_valid_utf8(pending));
        }

        if let Some(ref u2b) = self.gpt2_unicode_to_byte {
            // GPT-2: each char maps to a byte
            for ch in token_str.chars() {
                if let Some(&b) = u2b.get(&ch) {
                    pending.push(b);
                } else {
                    let mut buf = [0u8; 4];
                    let encoded = ch.encode_utf8(&mut buf);
                    pending.extend_from_slice(encoded.as_bytes());
                }
            }
            Ok(flush_valid_utf8(pending))
        } else {
            // SentencePiece: flush pending, then return token text
            let mut result = flush_valid_utf8(pending);
            result.push_str(&token_str.replace('▁', " "));
            Ok(result)
        }
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

    /// Load tokenizer from a HuggingFace `tokenizer.json` file
    ///
    /// This parses the JSON format used by HuggingFace tokenizers (the `tokenizers` library).
    /// Supports BPE models which cover LLaMA, Mistral, Qwen, and most modern LLMs.
    pub fn from_hf_json(path: impl AsRef<std::path::Path>) -> TokenizerResult<Self> {
        let path = path.as_ref();
        let data = std::fs::read_to_string(path)
            .map_err(|e| TokenizerError::MissingData(format!("{}: {}", path.display(), e)))?;

        Self::from_hf_json_str(&data)
    }

    /// Parse tokenizer from a HuggingFace tokenizer.json string
    pub fn from_hf_json_str(json: &str) -> TokenizerResult<Self> {
        let root: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| TokenizerError::EncodingError(format!("Invalid tokenizer.json: {}", e)))?;

        let model = root
            .get("model")
            .ok_or_else(|| TokenizerError::MissingData("model section in tokenizer.json".into()))?;

        let model_type = model.get("type").and_then(|v| v.as_str()).unwrap_or("BPE");
        let tokenizer_type = match model_type {
            "BPE" => TokenizerType::BPE,
            "Unigram" => TokenizerType::SentencePiece,
            "WordPiece" => TokenizerType::WordPiece,
            _ => TokenizerType::Unknown,
        };

        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();
        let mut scores = Vec::new();
        let mut merges = HashMap::new();
        let mut wordpiece_prefix = "##".to_string();
        let mut model_unk_token: Option<String> = None;

        match tokenizer_type {
            TokenizerType::SentencePiece => {
                // Unigram: vocab is [[token, score], ...]
                let vocab_arr = model
                    .get("vocab")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| {
                        TokenizerError::MissingData("Unigram vocab array".into())
                    })?;

                id_to_token = Vec::with_capacity(vocab_arr.len());
                scores = Vec::with_capacity(vocab_arr.len());

                for (id, entry) in vocab_arr.iter().enumerate() {
                    let arr = entry.as_array().ok_or_else(|| {
                        TokenizerError::MissingData(format!(
                            "Unigram vocab entry {} not an array",
                            id
                        ))
                    })?;
                    let token = arr
                        .first()
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| {
                            TokenizerError::MissingData(format!(
                                "Unigram vocab entry {} missing token",
                                id
                            ))
                        })?;
                    let score = arr
                        .get(1)
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0) as f32;

                    token_to_id.insert(token.to_string(), id as u32);
                    id_to_token.push(token.to_string());
                    scores.push(score);
                }

                if let Some(unk_id) = model.get("unk_id").and_then(|v| v.as_u64()) {
                    model_unk_token = id_to_token.get(unk_id as usize).cloned();
                }
            }
            TokenizerType::WordPiece => {
                // WordPiece: vocab is { token: id, ... }
                let vocab_obj = model
                    .get("vocab")
                    .and_then(|v| v.as_object())
                    .ok_or_else(|| {
                        TokenizerError::MissingData("WordPiece vocab object".into())
                    })?;

                let vocab_size = vocab_obj.len();
                id_to_token = vec![String::new(); vocab_size];

                for (token, id_val) in vocab_obj {
                    let id = id_val.as_u64().ok_or_else(|| {
                        TokenizerError::MissingData(format!("Invalid vocab ID for '{}'", token))
                    })? as u32;
                    token_to_id.insert(token.clone(), id);
                    if (id as usize) < id_to_token.len() {
                        id_to_token[id as usize] = token.clone();
                    }
                }

                if let Some(prefix) = model
                    .get("continuing_subword_prefix")
                    .and_then(|v| v.as_str())
                {
                    wordpiece_prefix = prefix.to_string();
                }
                if let Some(unk) = model.get("unk_token").and_then(|v| v.as_str()) {
                    model_unk_token = Some(unk.to_string());
                }

                scores = vec![0.0; id_to_token.len()];
            }
            _ => {
                // BPE: vocab is { token: id, ... }
                let vocab_obj = model
                    .get("vocab")
                    .and_then(|v| v.as_object())
                    .ok_or_else(|| {
                        TokenizerError::MissingData("BPE vocab object".into())
                    })?;

                let vocab_size = vocab_obj.len();
                id_to_token = vec![String::new(); vocab_size];

                for (token, id_val) in vocab_obj {
                    let id = id_val.as_u64().ok_or_else(|| {
                        TokenizerError::MissingData(format!("Invalid vocab ID for '{}'", token))
                    })? as u32;
                    token_to_id.insert(token.clone(), id);
                    if (id as usize) < id_to_token.len() {
                        id_to_token[id as usize] = token.clone();
                    }
                }

                if let Some(merges_arr) = model.get("merges").and_then(|v| v.as_array()) {
                    for (priority, merge_val) in merges_arr.iter().enumerate() {
                        if let Some(merge_str) = merge_val.as_str() {
                            let parts: Vec<&str> = merge_str.split(' ').collect();
                            if parts.len() == 2
                                && let (Some(&id1), Some(&id2)) =
                                    (token_to_id.get(parts[0]), token_to_id.get(parts[1]))
                            {
                                let merged = format!("{}{}", parts[0], parts[1]);
                                if let Some(&merged_id) = token_to_id.get(&merged) {
                                    merges.insert((id1, id2), (merged_id, priority));
                                }
                            }
                        }
                    }
                }

                scores = vec![0.0; id_to_token.len()];
            }
        }

        let vocab_size = id_to_token.len();

        // Parse added_tokens and detect special token roles
        let mut bos_token_id: Option<u32> = None;
        let mut eos_token_id: Option<u32> = None;
        let mut pad_token_id: Option<u32> = None;
        let mut unk_token_id: Option<u32> = None;

        if let Some(added_tokens) = root.get("added_tokens").and_then(|v| v.as_array()) {
            for token_obj in added_tokens {
                let content = token_obj
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let id = token_obj
                    .get("id")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32);
                let special = token_obj
                    .get("special")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                if let Some(id) = id {
                    token_to_id.insert(content.to_string(), id);
                    if (id as usize) < id_to_token.len() {
                        id_to_token[id as usize] = content.to_string();
                    }

                    if special {
                        let content_lower = content.to_lowercase();
                        if content_lower.contains("bos")
                            || content == "<s>"
                            || content == "<|begin_of_text|>"
                            || content == "<|startoftext|>"
                        {
                            bos_token_id = Some(id);
                        }
                        if content_lower.contains("eos")
                            || content == "</s>"
                            || content == "<|end_of_text|>"
                            || content == "<|endoftext|>"
                            || content == "<|eot_id|>"
                        {
                            eos_token_id = Some(id);
                        }
                        if content_lower.contains("pad") || content == "<pad>" {
                            pad_token_id = Some(id);
                        }
                        if content_lower.contains("unk") || content == "<unk>" {
                            unk_token_id = Some(id);
                        }
                    }
                }
            }
        }

        // Resolve unk from model section if not found in added_tokens
        if unk_token_id.is_none() {
            if let Some(ref unk_str) = model_unk_token {
                unk_token_id = token_to_id.get(unk_str).copied();
            }
        }

        // Check post_processor for special token IDs
        if let Some(post_proc) = root.get("post_processor") {
            if let Some(special_tokens_map) = post_proc.get("special_tokens") {
                if let Some(bos_obj) = special_tokens_map
                    .get("<s>")
                    .or_else(|| special_tokens_map.get("<|begin_of_text|>"))
                    && let Some(ids) = bos_obj.get("ids").and_then(|v| v.as_array())
                    && let Some(id) = ids.first().and_then(|v| v.as_u64())
                {
                    bos_token_id = bos_token_id.or(Some(id as u32));
                }
                if let Some(eos_obj) = special_tokens_map
                    .get("</s>")
                    .or_else(|| special_tokens_map.get("<|end_of_text|>"))
                    && let Some(ids) = eos_obj.get("ids").and_then(|v| v.as_array())
                    && let Some(id) = ids.first().and_then(|v| v.as_u64())
                {
                    eos_token_id = eos_token_id.or(Some(id as u32));
                }
            }
        }

        let special_tokens = SpecialTokens {
            bos_token_id: bos_token_id.unwrap_or(1),
            eos_token_id: eos_token_id.unwrap_or(2),
            pad_token_id,
            unk_token_id,
        };

        // Build token types
        let mut token_types = vec![TokenType::Normal; vocab_size];
        for &id in [special_tokens.bos_token_id, special_tokens.eos_token_id].iter() {
            if (id as usize) < token_types.len() {
                token_types[id as usize] = TokenType::Control;
            }
        }
        if let Some(pad_id) = special_tokens.pad_token_id
            && (pad_id as usize) < token_types.len()
        {
            token_types[pad_id as usize] = TokenType::Control;
        }
        if let Some(unk_id) = special_tokens.unk_token_id
            && (unk_id as usize) < token_types.len()
        {
            token_types[unk_id as usize] = TokenType::Control;
        }
        for (token, &id) in &token_to_id {
            if token.starts_with("<0x")
                && token.ends_with('>')
                && token.len() == 6
                && (id as usize) < token_types.len()
            {
                token_types[id as usize] = TokenType::Byte;
            }
        }

        // Detect GPT-2 byte-level BPE
        let uses_byte_level = root
            .get("pre_tokenizer")
            .and_then(|v| v.get("type").or_else(|| {
                // Handle Sequence pre-tokenizer containing ByteLevel
                v.get("pretokenizers").and_then(|arr| {
                    arr.as_array().and_then(|a| {
                        a.iter().find_map(|pt| {
                            pt.get("type").filter(|t| t.as_str() == Some("ByteLevel"))
                        })
                    })
                })
            }))
            .and_then(|v| v.as_str())
            .is_some_and(|t| t == "ByteLevel");

        let (gpt2_unicode_to_byte, gpt2_byte_to_unicode) = if tokenizer_type == TokenizerType::BPE && uses_byte_level {
            let (u2b, b2u) = build_gpt2_mappings();
            (Some(u2b), Some(b2u))
        } else {
            (None, None)
        };

        // Parse HF pipeline components
        let normalizer = root.get("normalizer")
            .and_then(|v| if v.is_null() { None } else { Self::parse_normalizer(v) });
        let pre_tokenizer = root.get("pre_tokenizer")
            .and_then(|v| if v.is_null() { None } else { Self::parse_pre_tokenizer(v) });
        let post_processor = root.get("post_processor")
            .and_then(|v| if v.is_null() { None } else { Self::parse_post_processor(v, &token_to_id) });

        let mut control_token_strings: Vec<(String, u32)> = token_types
            .iter()
            .enumerate()
            .filter(|(_, tt)| **tt == TokenType::Control)
            .filter_map(|(id, _)| {
                let s = &id_to_token[id];
                if !s.is_empty() {
                    Some((s.clone(), id as u32))
                } else {
                    None
                }
            })
            .collect();
        control_token_strings.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        Ok(Self {
            token_to_id,
            id_to_token,
            scores,
            merges,
            special_tokens,
            tokenizer_type,
            vocab_size,
            token_types,
            gpt2_unicode_to_byte,
            gpt2_byte_to_unicode,
            normalizer,
            pre_tokenizer,
            post_processor,
            wordpiece_prefix,
            control_token_strings,
            has_explicit_bos: bos_token_id.is_some(),
        })
    }

    fn parse_normalizer(value: &serde_json::Value) -> Option<Normalizer> {
        let type_str = value.get("type")?.as_str()?;
        match type_str {
            "NFC" => Some(Normalizer::NFC),
            "NFKC" => Some(Normalizer::NFKC),
            "NFD" => Some(Normalizer::NFD),
            "NFKD" => Some(Normalizer::NFKD),
            "Lowercase" => Some(Normalizer::Lowercase),
            "Strip" => {
                let left = value.get("strip_left").and_then(|v| v.as_bool()).unwrap_or(true);
                let right = value.get("strip_right").and_then(|v| v.as_bool()).unwrap_or(true);
                Some(Normalizer::Strip { left, right })
            }
            "Prepend" => {
                let prepend = value.get("prepend").and_then(|v| v.as_str()).unwrap_or("▁");
                Some(Normalizer::Prepend(prepend.to_string()))
            }
            "Replace" => {
                let pattern = value
                    .get("pattern")
                    .and_then(|v| v.get("String").and_then(|s| s.as_str()))
                    .unwrap_or("");
                let content = value.get("content").and_then(|v| v.as_str()).unwrap_or("");
                Some(Normalizer::Replace {
                    pattern: pattern.to_string(),
                    content: content.to_string(),
                })
            }
            "StripAccents" => Some(Normalizer::StripAccents),
            "Sequence" => {
                let normalizers = value.get("normalizers")?.as_array()?;
                let parsed: Vec<Normalizer> = normalizers
                    .iter()
                    .filter_map(|v| Self::parse_normalizer(v))
                    .collect();
                if parsed.is_empty() {
                    None
                } else {
                    Some(Normalizer::Sequence(parsed))
                }
            }
            "BertNormalizer" => {
                let mut seq = Vec::new();
                if value
                    .get("lowercase")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true)
                {
                    seq.push(Normalizer::Lowercase);
                }
                if value
                    .get("strip_accents")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    seq.push(Normalizer::StripAccents);
                }
                match seq.len() {
                    0 => None,
                    1 => Some(seq.remove(0)),
                    _ => Some(Normalizer::Sequence(seq)),
                }
            }
            "Precompiled" => Some(Normalizer::NFC),
            _ => None,
        }
    }

    fn parse_pre_tokenizer(value: &serde_json::Value) -> Option<PreTokenizer> {
        let type_str = value.get("type")?.as_str()?;
        match type_str {
            "ByteLevel" => {
                let add_prefix_space = value
                    .get("add_prefix_space")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                Some(PreTokenizer::ByteLevel { add_prefix_space })
            }
            "Whitespace" | "WhitespaceSplit" => Some(PreTokenizer::Whitespace),
            "Metaspace" => {
                let replacement = value
                    .get("replacement")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.chars().next())
                    .unwrap_or('▁');
                let add_prefix_space = value
                    .get("add_prefix_space")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                Some(PreTokenizer::Metaspace {
                    replacement,
                    add_prefix_space,
                })
            }
            "Punctuation" | "BertPreTokenizer" => Some(PreTokenizer::Punctuation),
            "Digits" => {
                let individual_digits = value
                    .get("individual_digits")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                Some(PreTokenizer::Digits { individual_digits })
            }
            "Sequence" => {
                let pretokenizers = value.get("pretokenizers")?.as_array()?;
                let parsed: Vec<PreTokenizer> = pretokenizers
                    .iter()
                    .filter_map(|v| Self::parse_pre_tokenizer(v))
                    .collect();
                if parsed.is_empty() {
                    None
                } else {
                    Some(PreTokenizer::Sequence(parsed))
                }
            }
            _ => None,
        }
    }

    fn parse_post_processor(
        value: &serde_json::Value,
        token_to_id: &HashMap<String, u32>,
    ) -> Option<PostProcessor> {
        let type_str = value.get("type")?.as_str()?;
        match type_str {
            "TemplateProcessing" => {
                let parse_template = |arr: &[serde_json::Value]| -> Vec<TemplateElement> {
                    arr.iter()
                        .filter_map(|item| {
                            if let Some(special) = item.get("SpecialToken") {
                                let id_str = special.get("id")?.as_str()?;
                                let token_id = token_to_id.get(id_str).copied()?;
                                Some(TemplateElement::SpecialToken {
                                    id: id_str.to_string(),
                                    token_id,
                                })
                            } else if item.get("Sequence").is_some() {
                                let type_id = item
                                    .get("Sequence")
                                    .and_then(|s| s.get("id"))
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0) as u32;
                                Some(TemplateElement::Sequence { type_id })
                            } else {
                                None
                            }
                        })
                        .collect()
                };

                let single = value
                    .get("single")
                    .and_then(|v| v.as_array())
                    .map(|a| parse_template(a))
                    .unwrap_or_default();
                let pair = value
                    .get("pair")
                    .and_then(|v| v.as_array())
                    .map(|a| parse_template(a))
                    .unwrap_or_default();

                Some(PostProcessor::TemplateProcessing { single, pair })
            }
            "ByteLevel" => {
                let trim_offsets = value
                    .get("trim_offsets")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                Some(PostProcessor::ByteLevel { trim_offsets })
            }
            "BertProcessing" => {
                let mut single = Vec::new();
                let mut pair = Vec::new();

                if let Some(cls) = value.get("cls").and_then(|v| v.as_array()) {
                    if let (Some(token), Some(id)) = (
                        cls.first().and_then(|v| v.as_str()),
                        cls.get(1).and_then(|v| v.as_u64()),
                    ) {
                        let elem = TemplateElement::SpecialToken {
                            id: token.to_string(),
                            token_id: id as u32,
                        };
                        single.push(elem.clone());
                        pair.push(elem);
                    }
                }

                single.push(TemplateElement::Sequence { type_id: 0 });
                pair.push(TemplateElement::Sequence { type_id: 0 });

                if let Some(sep) = value.get("sep").and_then(|v| v.as_array()) {
                    if let (Some(token), Some(id)) = (
                        sep.first().and_then(|v| v.as_str()),
                        sep.get(1).and_then(|v| v.as_u64()),
                    ) {
                        let elem = TemplateElement::SpecialToken {
                            id: token.to_string(),
                            token_id: id as u32,
                        };
                        single.push(elem.clone());
                        pair.push(elem.clone());
                        pair.push(TemplateElement::Sequence { type_id: 1 });
                        pair.push(elem);
                    }
                }

                Some(PostProcessor::TemplateProcessing { single, pair })
            }
            _ => None,
        }
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

    #[test]
    fn test_gpt2_unicode_to_byte_table() {
        let table = build_gpt2_unicode_to_byte();
        assert_eq!(table.len(), 256);

        // Printable ASCII maps to itself
        assert_eq!(table[&'!'], b'!');
        assert_eq!(table[&'A'], b'A');
        assert_eq!(table[&'~'], b'~');

        // GPT-2 special chars map to their byte values
        assert_eq!(table[&'Ġ'], b' '); // U+0120 → 0x20 (space)
        assert_eq!(table[&'Ċ'], b'\n'); // U+010A → 0x0A (newline)
        assert_eq!(table[&'ĉ'], b'\t'); // U+0109 → 0x09 (tab)

        // Latin-1 supplement bytes map to themselves
        assert_eq!(table[&'¡'], 0xA1);
        assert_eq!(table[&'®'], 0xAE);
        assert_eq!(table[&'ÿ'], 0xFF);
    }

    #[test]
    fn test_gpt2_decode_space_and_emoji() {
        let table = build_gpt2_unicode_to_byte();

        // "ĠHello" should decode to " Hello"
        let bytes: Vec<u8> = "ĠHello".chars().map(|c| table[&c]).collect();
        assert_eq!(String::from_utf8(bytes).unwrap(), " Hello");

        // "ðŁĺĬ" is the GPT-2 encoding of 😊 (U+1F60A, UTF-8: F0 9F 98 8A)
        let bytes: Vec<u8> = "ðŁĺĬ".chars().map(|c| table[&c]).collect();
        let decoded = String::from_utf8(bytes).unwrap();
        assert_eq!(decoded, "😊");
    }

    #[test]
    fn test_normalizer_nfc() {
        let norm = Normalizer::NFC;
        // U+00E9 (é precomposed) vs U+0065 + U+0301 (e + combining acute)
        let decomposed = "e\u{0301}";
        let result = norm.apply(decomposed);
        assert_eq!(result, "\u{00E9}");
    }

    #[test]
    fn test_normalizer_lowercase() {
        let norm = Normalizer::Lowercase;
        assert_eq!(norm.apply("HELLO World"), "hello world");
    }

    #[test]
    fn test_normalizer_strip_accents() {
        let norm = Normalizer::StripAccents;
        assert_eq!(norm.apply("café"), "cafe");
        assert_eq!(norm.apply("naïve"), "naive");
    }

    #[test]
    fn test_normalizer_sequence() {
        let norm = Normalizer::Sequence(vec![
            Normalizer::NFKC,
            Normalizer::Lowercase,
        ]);
        assert_eq!(norm.apply("HÉLLO"), "héllo");
    }

    #[test]
    fn test_normalizer_replace() {
        let norm = Normalizer::Replace {
            pattern: " ".to_string(),
            content: "▁".to_string(),
        };
        assert_eq!(norm.apply("hello world"), "hello▁world");
    }

    #[test]
    fn test_pre_tokenizer_whitespace() {
        let pt = PreTokenizer::Whitespace;
        assert_eq!(pt.apply("Hello world  test"), vec!["Hello", "world", "test"]);
    }

    #[test]
    fn test_pre_tokenizer_byte_level() {
        let pt = PreTokenizer::ByteLevel { add_prefix_space: true };
        let result = pt.apply("Hello world");
        assert_eq!(result, vec![" Hello", " world"]);

        let pt_no_space = PreTokenizer::ByteLevel { add_prefix_space: false };
        let result = pt_no_space.apply("Hello world");
        assert_eq!(result, vec!["Hello", " world"]);
    }

    #[test]
    fn test_pre_tokenizer_punctuation() {
        let pt = PreTokenizer::Punctuation;
        let result = pt.apply("Hello, world!");
        assert_eq!(result, vec!["Hello", ",", " world", "!"]);
    }

    #[test]
    fn test_pre_tokenizer_digits() {
        let pt = PreTokenizer::Digits { individual_digits: true };
        let result = pt.apply("abc123def");
        assert_eq!(result, vec!["abc", "1", "2", "3", "def"]);
    }

    #[test]
    fn test_pre_tokenizer_sequence() {
        let pt = PreTokenizer::Sequence(vec![
            PreTokenizer::Whitespace,
            PreTokenizer::Punctuation,
        ]);
        let result = pt.apply("Hello, world!");
        assert_eq!(result, vec!["Hello", ",", "world", "!"]);
    }

    #[test]
    fn test_unigram_from_hf_json() {
        let json = r#"{
            "model": {
                "type": "Unigram",
                "unk_id": 0,
                "vocab": [
                    ["<unk>", 0.0],
                    ["▁", -1.0],
                    ["▁the", -2.0],
                    ["▁a", -2.5],
                    ["h", -3.0],
                    ["e", -3.0],
                    ["l", -3.0],
                    ["o", -3.0],
                    ["he", -2.0],
                    ["llo", -2.5]
                ]
            },
            "pre_tokenizer": {
                "type": "Metaspace",
                "replacement": "▁",
                "add_prefix_space": true
            },
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true}
            ]
        }"#;

        let tok = Tokenizer::from_hf_json_str(json).unwrap();
        assert_eq!(tok.tokenizer_type, TokenizerType::SentencePiece);
        assert_eq!(tok.vocab_size, 10);
        assert!(tok.scores.iter().any(|&s| s != 0.0));
    }

    #[test]
    fn test_wordpiece_from_hf_json() {
        let json = r###"{
            "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "vocab": {
                    "[UNK]": 0,
                    "[CLS]": 1,
                    "[SEP]": 2,
                    "hello": 3,
                    "world": 4,
                    "he": 5,
                    "##llo": 6,
                    "wo": 7,
                    "##rld": 8
                }
            },
            "normalizer": {
                "type": "BertNormalizer",
                "lowercase": true,
                "strip_accents": false
            },
            "pre_tokenizer": {
                "type": "BertPreTokenizer"
            },
            "added_tokens": [
                {"id": 0, "content": "[UNK]", "special": true},
                {"id": 1, "content": "[CLS]", "special": true},
                {"id": 2, "content": "[SEP]", "special": true}
            ]
        }"###;

        let tok = Tokenizer::from_hf_json_str(json).unwrap();
        assert_eq!(tok.tokenizer_type, TokenizerType::WordPiece);
        assert_eq!(tok.wordpiece_prefix, "##");

        // "hello" should encode to [3] (direct match)
        let tokens = tok.encode("hello", false).unwrap();
        assert_eq!(tokens, vec![3]);

        // "hello world" should encode to [3, 4] (both direct matches after whitespace split)
        let tokens = tok.encode("hello world", false).unwrap();
        assert_eq!(tokens, vec![3, 4]);
    }

    #[test]
    fn test_wordpiece_subword_splitting() {
        let json = r###"{
            "model": {
                "type": "WordPiece",
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "vocab": {
                    "[UNK]": 0,
                    "un": 1,
                    "##know": 2,
                    "##n": 3,
                    "unknown": 4,
                    "the": 5,
                    "##s": 6
                }
            },
            "pre_tokenizer": { "type": "Whitespace" },
            "added_tokens": [
                {"id": 0, "content": "[UNK]", "special": true}
            ]
        }"###;

        let tok = Tokenizer::from_hf_json_str(json).unwrap();

        // "unknown" is a direct vocabulary match
        let tokens = tok.encode("unknown", false).unwrap();
        assert_eq!(tokens, vec![4]);

        // "the" should encode to [5]
        let tokens = tok.encode("the", false).unwrap();
        assert_eq!(tokens, vec![5]);

        // "thes" should split to "the" + "##s"
        let tokens = tok.encode("thes", false).unwrap();
        assert_eq!(tokens, vec![5, 6]);
    }

    #[test]
    fn test_unigram_viterbi_encoding() {
        let json = r#"{
            "model": {
                "type": "Unigram",
                "unk_id": 0,
                "vocab": [
                    ["<unk>", 0.0],
                    ["a", -1.0],
                    ["b", -1.0],
                    ["c", -1.0],
                    ["ab", -0.5],
                    ["bc", -0.5],
                    ["abc", -0.1]
                ]
            },
            "pre_tokenizer": { "type": "Whitespace" },
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true}
            ]
        }"#;

        let tok = Tokenizer::from_hf_json_str(json).unwrap();

        // "abc" should prefer the single token [abc] (score -0.1) over
        // [a,bc] (score -1.5) or [ab,c] (score -1.5) or [a,b,c] (score -3.0)
        let tokens = tok.encode("abc", false).unwrap();
        assert_eq!(tokens, vec![6]); // id 6 = "abc"
    }

    #[test]
    fn test_bpe_with_pipeline() {
        let json = r#"{
            "model": {
                "type": "BPE",
                "vocab": {
                    "h": 0,
                    "e": 1,
                    "l": 2,
                    "o": 3,
                    "he": 4,
                    "ll": 5,
                    "hello": 6,
                    " ": 7
                },
                "merges": [
                    "h e",
                    "l l",
                    "he ll",
                    "hell o"
                ]
            },
            "pre_tokenizer": {
                "type": "ByteLevel",
                "add_prefix_space": false
            },
            "added_tokens": []
        }"#;

        let tok = Tokenizer::from_hf_json_str(json).unwrap();
        assert_eq!(tok.tokenizer_type, TokenizerType::BPE);
        assert!(tok.pre_tokenizer.is_some());

        // Should encode "hello" -> merge h+e=he, l+l=ll, he+ll=hell, hell+o=hello -> [6]
        let tokens = tok.encode("hello", false).unwrap();
        assert_eq!(tokens, vec![6]);
    }

    #[test]
    fn test_parse_normalizer_types() {
        let nfc: serde_json::Value = serde_json::from_str(r#"{"type": "NFC"}"#).unwrap();
        let result = Tokenizer::parse_normalizer(&nfc);
        assert!(matches!(result, Some(Normalizer::NFC)));

        let bert: serde_json::Value = serde_json::from_str(
            r#"{"type": "BertNormalizer", "lowercase": true, "strip_accents": true}"#,
        )
        .unwrap();
        let result = Tokenizer::parse_normalizer(&bert);
        assert!(matches!(result, Some(Normalizer::Sequence(_))));

        let seq: serde_json::Value = serde_json::from_str(
            r#"{"type": "Sequence", "normalizers": [{"type": "NFC"}, {"type": "Lowercase"}]}"#,
        )
        .unwrap();
        let result = Tokenizer::parse_normalizer(&seq);
        assert!(matches!(result, Some(Normalizer::Sequence(_))));
    }

    #[test]
    fn test_parse_pre_tokenizer_types() {
        let bl: serde_json::Value =
            serde_json::from_str(r#"{"type": "ByteLevel", "add_prefix_space": false}"#).unwrap();
        let result = Tokenizer::parse_pre_tokenizer(&bl);
        assert!(matches!(
            result,
            Some(PreTokenizer::ByteLevel { add_prefix_space: false })
        ));

        let meta: serde_json::Value = serde_json::from_str(
            r#"{"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}"#,
        )
        .unwrap();
        let result = Tokenizer::parse_pre_tokenizer(&meta);
        assert!(matches!(
            result,
            Some(PreTokenizer::Metaspace { add_prefix_space: true, .. })
        ));

        let seq: serde_json::Value = serde_json::from_str(
            r#"{"type": "Sequence", "pretokenizers": [{"type": "Whitespace"}, {"type": "Punctuation"}]}"#,
        )
        .unwrap();
        let result = Tokenizer::parse_pre_tokenizer(&seq);
        assert!(matches!(result, Some(PreTokenizer::Sequence(_))));
    }
}
