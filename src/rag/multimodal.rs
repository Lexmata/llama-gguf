//! Multi-modal content support for RAG
//!
//! Provides content type detection, image/table extraction from documents,
//! and a framework for multi-modal embedding generation.

#![cfg(any(feature = "rag", feature = "rag-sqlite"))]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::RagResult;

/// Content types supported by the RAG system
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentType {
    #[default]
    Text,
    Image,
    Table,
    Code,
    Mixed,
}

/// A content element extracted from a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentElement {
    pub content_type: ContentType,
    pub text: String,
    pub source_uri: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Multi-modal document with typed content elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalDocument {
    pub source_uri: String,
    pub elements: Vec<ContentElement>,
    pub raw_text: String,
}

/// Configuration for content extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    pub extract_tables: bool,
    pub extract_code_blocks: bool,
    pub extract_image_refs: bool,
    pub table_format: TableFormat,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TableFormat {
    #[default]
    Markdown,
    Csv,
    Plain,
}

/// Extracts structured content elements from raw text
pub struct ContentExtractor {
    config: ExtractionConfig,
}

impl ContentExtractor {
    pub fn new(config: ExtractionConfig) -> Self {
        Self { config }
    }

    /// Extract content elements from markdown text
    pub fn extract_markdown(&self, text: &str, source_uri: Option<&str>) -> MultiModalDocument {
        let source_uri = source_uri.unwrap_or("").to_string();
        let mut elements = Vec::new();
        let mut i = 0;
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();

        while i < n {
            // Try code block first (```...```)
            if self.config.extract_code_blocks && i + 3 <= n && &chars[i..i + 3].iter().collect::<String>() == "```" {
                let (element, advance) = parse_code_block(&chars[i..], &source_uri);
                elements.push(element);
                i += advance;
                continue;
            }

            // Try image reference ![alt](url)
            if self.config.extract_image_refs && i + 2 <= n && chars[i] == '!' && chars.get(i + 1) == Some(&'[') {
                let (element, advance) = parse_image_ref(&chars[i..], &source_uri);
                elements.push(element);
                i += advance;
                continue;
            }

            // Try table (lines with |...|)
            if self.config.extract_tables {
                let line_start = i;
                let line_end = chars[i..].iter().position(|&c| c == '\n').map(|p| i + p).unwrap_or(n);
                let line: String = chars[line_start..line_end].iter().collect();
                if is_table_line(&line) {
                    let (table_lines, advance) = collect_table_lines(&chars[i..]);
                    if table_lines.len() >= 2 {
                        let (element, _) = parse_table(&table_lines, &source_uri);
                        elements.push(element);
                        i += advance;
                        continue;
                    }
                }
            }

            // Regular text - advance to next special pattern or end
            let text_start = i;
            let mut j = i;
            while j < n {
                // Check for code block start
                if j + 3 <= n && chars[j] == '`' && chars.get(j + 1) == Some(&'`') && chars.get(j + 2) == Some(&'`') {
                    break;
                }
                // Check for image start
                if j + 2 <= n && chars[j] == '!' && chars.get(j + 1) == Some(&'[') {
                    break;
                }
                // Check for table (start of line with |)
                if j > 0 && chars[j - 1] == '\n' && chars[j] == '|' {
                    break;
                }
                j += 1;
            }
            let text_content: String = chars[text_start..j].iter().collect();
            let trimmed = text_content.trim();
            if !trimmed.is_empty() {
                let mut metadata = HashMap::new();
                metadata.insert("char_count".to_string(), serde_json::json!(trimmed.len()));
                elements.push(ContentElement {
                    content_type: ContentType::Text,
                    text: trimmed.to_string(),
                    source_uri: if source_uri.is_empty() { None } else { Some(source_uri.clone()) },
                    metadata,
                });
            }
            i = j;
        }

        MultiModalDocument {
            source_uri: source_uri.clone(),
            elements,
            raw_text: text.to_string(),
        }
    }

    /// Extract content elements from plain text
    pub fn extract_plain(&self, text: &str, source_uri: Option<&str>) -> MultiModalDocument {
        let source_uri = source_uri.unwrap_or("").to_string();
        let trimmed = text.trim();
        let mut metadata = HashMap::new();
        metadata.insert("char_count".to_string(), serde_json::json!(trimmed.len()));
        let elements = if trimmed.is_empty() {
            vec![]
        } else {
            vec![ContentElement {
                content_type: ContentType::Text,
                text: trimmed.to_string(),
                source_uri: if source_uri.is_empty() { None } else { Some(source_uri.clone()) },
                metadata,
            }]
        };
        MultiModalDocument {
            source_uri: source_uri.clone(),
            elements,
            raw_text: text.to_string(),
        }
    }

    /// Detect content type from text
    pub fn detect_content_type(text: &str) -> ContentType {
        let has_table = text.lines().any(|l| is_table_line(l));
        let has_code = text.contains("```");
        let has_image = text.contains("![") && text.contains("](");
        // "Other text" = content that isn't solely table, code block, or image
        let all_table = !text.trim().is_empty()
            && text.lines().all(|l| l.trim().is_empty() || is_table_line(l));
        let solely_code_block = {
            let t = text.trim();
            t.starts_with("```") && t.ends_with("```") && t.matches("```").count() == 2
        };
        let non_empty_lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        let solely_image = has_image && !has_table && !has_code && non_empty_lines.len() <= 1;
        let has_other_text =
            !text.trim().is_empty() && !all_table && !solely_code_block && !solely_image;

        let mut count = 0;
        if has_table { count += 1; }
        if has_code { count += 1; }
        if has_image { count += 1; }
        if has_other_text { count += 1; }

        if count > 1 {
            ContentType::Mixed
        } else if has_table {
            ContentType::Table
        } else if has_code {
            ContentType::Code
        } else if has_image {
            ContentType::Image
        } else {
            ContentType::Text
        }
    }
}

impl Default for ContentExtractor {
    fn default() -> Self {
        Self {
            config: ExtractionConfig {
                extract_tables: true,
                extract_code_blocks: true,
                extract_image_refs: true,
                table_format: TableFormat::default(),
            },
        }
    }
}

fn is_table_line(line: &str) -> bool {
    let trimmed = line.trim();
    if !trimmed.starts_with('|') || !trimmed.ends_with('|') {
        return false;
    }
    trimmed.contains('|') && trimmed.len() > 2
}

fn parse_code_block(chars: &[char], source_uri: &str) -> (ContentElement, usize) {
    let mut i = 3; // skip ```
    let mut language = String::new();
    while i < chars.len() && chars[i] != '\n' && chars[i] != '`' {
        language.push(chars[i]);
        i += 1;
    }
    let language = language.trim().to_string();
    if i < chars.len() && chars[i] == '\n' {
        i += 1;
    }
    let content_start = i;
    let mut j = i;
    while j + 3 <= chars.len() {
        if chars[j] == '`' && chars[j + 1] == '`' && chars[j + 2] == '`' {
            break;
        }
        j += 1;
    }
    let content: String = chars[content_start..j].iter().collect();
    let line_count = content.lines().count();
    let advance = (j + 3).min(chars.len());

    let mut metadata = HashMap::new();
    metadata.insert("language".to_string(), serde_json::json!(language));
    metadata.insert("line_count".to_string(), serde_json::json!(line_count));

    let element = ContentElement {
        content_type: ContentType::Code,
        text: content,
        source_uri: if source_uri.is_empty() { None } else { Some(source_uri.to_string()) },
        metadata,
    };
    (element, advance)
}

fn parse_image_ref(chars: &[char], source_uri: &str) -> (ContentElement, usize) {
    if chars.len() < 4 || chars[0] != '!' || chars[1] != '[' {
        return (ContentElement {
            content_type: ContentType::Image,
            text: String::new(),
            source_uri: None,
            metadata: HashMap::new(),
        }, 0);
    }
    let mut i = 2;
    let mut alt = String::new();
    while i < chars.len() && chars[i] != ']' {
        alt.push(chars[i]);
        i += 1;
    }
    if i >= chars.len() || chars[i] != ']' {
        return (ContentElement {
            content_type: ContentType::Image,
            text: String::new(),
            source_uri: None,
            metadata: HashMap::new(),
        }, 2);
    }
    i += 1;
    if i >= chars.len() || chars[i] != '(' {
        return (ContentElement {
            content_type: ContentType::Image,
            text: alt,
            source_uri: None,
            metadata: HashMap::new(),
        }, i);
    }
    i += 1;
    let url_start = i;
    while i < chars.len() && chars[i] != ')' {
        i += 1;
    }
    let url: String = chars[url_start..i].iter().collect();
    let advance = if i < chars.len() { i + 1 } else { i };

    let mut metadata = HashMap::new();
    metadata.insert("alt_text".to_string(), serde_json::json!(alt));
    metadata.insert("url".to_string(), serde_json::json!(url));

    let element = ContentElement {
        content_type: ContentType::Image,
        text: url.clone(),
        source_uri: if source_uri.is_empty() { None } else { Some(source_uri.to_string()) },
        metadata,
    };
    (element, advance)
}

fn collect_table_lines(chars: &[char]) -> (Vec<String>, usize) {
    let mut lines = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let line_end = chars[i..].iter().position(|&c| c == '\n').map(|p| i + p).unwrap_or(chars.len());
        let line: String = chars[i..line_end].iter().collect();
        if !is_table_line(&line) {
            break;
        }
        lines.push(line);
        i = if line_end < chars.len() { line_end + 1 } else { line_end };
    }
    let advance = i;
    (lines, advance)
}

fn is_separator_line(line: &str) -> bool {
    let trimmed = line.trim();
    if !trimmed.starts_with('|') || !trimmed.ends_with('|') {
        return false;
    }
    trimmed[1..trimmed.len() - 1]
        .split('|')
        .all(|cell| cell.trim().chars().all(|c| c == '-' || c == ':'))
}

fn parse_table(lines: &[String], source_uri: &str) -> (ContentElement, usize) {
    let table_text = lines.join("\n");
    let rows: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
    let row_count = rows.iter().filter(|l| !is_separator_line(l)).count();
    let col_count = rows.first().map(|r| r.split('|').filter(|s| !s.trim().is_empty()).count()).unwrap_or(0);
    let header = rows.first().map(|r| r.to_string()).unwrap_or_default();

    let mut metadata = HashMap::new();
    metadata.insert("rows".to_string(), serde_json::json!(row_count));
    metadata.insert("cols".to_string(), serde_json::json!(col_count));
    metadata.insert("header".to_string(), serde_json::json!(header));

    let element = ContentElement {
        content_type: ContentType::Table,
        text: table_text,
        source_uri: if source_uri.is_empty() { None } else { Some(source_uri.to_string()) },
        metadata,
    };
    (element, 0)
}

/// Trait for generating embeddings from different content types
pub trait EmbeddingProvider: Send + Sync {
    /// Generate embedding for text content
    fn embed_text(&self, text: &str) -> RagResult<Vec<f32>>;

    /// Generate embedding for an image (returns None if not supported)
    fn embed_image(&self, _image_data: &[u8]) -> RagResult<Option<Vec<f32>>> {
        Ok(None) // Default: not supported
    }

    /// Get the embedding dimension
    fn dimension(&self) -> usize;

    /// Supported content types
    fn supported_types(&self) -> Vec<ContentType> {
        vec![ContentType::Text]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_type_default() {
        assert_eq!(ContentType::default(), ContentType::Text);
    }

    #[test]
    fn test_extract_markdown_table() {
        let extractor = ContentExtractor::default();
        let md = r#"| a | b |
|---|---|
| 1 | 2 |"#;
        let doc = extractor.extract_markdown(md, Some("test.md"));
        assert_eq!(doc.elements.len(), 1);
        assert_eq!(doc.elements[0].content_type, ContentType::Table);
        assert!(doc.elements[0].text.contains("|"));
        assert_eq!(doc.elements[0].metadata.get("rows").and_then(|v| v.as_u64()), Some(2));
        assert_eq!(doc.elements[0].metadata.get("cols").and_then(|v| v.as_u64()), Some(2));
    }

    #[test]
    fn test_extract_code_block() {
        let extractor = ContentExtractor::default();
        let md = r#"```python
def hello():
    print("world")
```"#;
        let doc = extractor.extract_markdown(md, None);
        assert_eq!(doc.elements.len(), 1);
        assert_eq!(doc.elements[0].content_type, ContentType::Code);
        assert!(doc.elements[0].text.contains("def hello"));
        assert_eq!(doc.elements[0].metadata.get("language").and_then(|v| v.as_str()), Some("python"));
        assert_eq!(doc.elements[0].metadata.get("line_count").and_then(|v| v.as_u64()), Some(2));
    }

    #[test]
    fn test_extract_image_ref() {
        let extractor = ContentExtractor::default();
        let md = r#"![alt text](https://example.com/image.png)"#;
        let doc = extractor.extract_markdown(md, Some("doc.md"));
        assert_eq!(doc.elements.len(), 1);
        assert_eq!(doc.elements[0].content_type, ContentType::Image);
        assert_eq!(doc.elements[0].text, "https://example.com/image.png");
        assert_eq!(doc.elements[0].metadata.get("alt_text").and_then(|v| v.as_str()), Some("alt text"));
        assert_eq!(doc.elements[0].metadata.get("url").and_then(|v| v.as_str()), Some("https://example.com/image.png"));
    }

    #[test]
    fn test_extract_mixed_content() {
        let extractor = ContentExtractor::default();
        let md = r#"# Title

Some intro text.

| col |
|---|
| x |

```rust
fn main() {}
```

More text."#;
        let doc = extractor.extract_markdown(md, Some("mixed.md"));
        assert!(doc.elements.len() >= 3);
        let types: Vec<ContentType> = doc.elements.iter().map(|e| e.content_type).collect();
        assert!(types.contains(&ContentType::Text));
        assert!(types.contains(&ContentType::Table));
        assert!(types.contains(&ContentType::Code));
    }

    #[test]
    fn test_detect_content_type() {
        assert_eq!(ContentExtractor::detect_content_type("plain text"), ContentType::Text);
        assert_eq!(ContentExtractor::detect_content_type("|a|b|\n|---|\n|1|2|"), ContentType::Table);
        assert_eq!(ContentExtractor::detect_content_type("```\ncode\n```"), ContentType::Code);
        assert_eq!(ContentExtractor::detect_content_type("![x](y)"), ContentType::Image);
        assert_eq!(ContentExtractor::detect_content_type("text\n|a|b|\n```x```"), ContentType::Mixed);
    }

    #[test]
    fn test_multimodal_document() {
        let extractor = ContentExtractor::default();
        let doc = extractor.extract_markdown("hello", Some("x.txt"));
        assert_eq!(doc.source_uri, "x.txt");
        assert_eq!(doc.raw_text, "hello");
        assert_eq!(doc.elements.len(), 1);
        assert_eq!(doc.elements[0].content_type, ContentType::Text);
        assert_eq!(doc.elements[0].text, "hello");
    }

    #[test]
    fn test_extraction_config_defaults() {
        let extractor = ContentExtractor::default();
        assert!(extractor.config.extract_tables);
        assert!(extractor.config.extract_code_blocks);
        assert!(extractor.config.extract_image_refs);
        assert_eq!(extractor.config.table_format, TableFormat::Markdown);
    }
}
