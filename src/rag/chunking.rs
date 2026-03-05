//! Advanced document chunking strategies for RAG
//!
//! Provides multiple strategies for splitting documents into chunks
//! suitable for embedding and retrieval.

use serde::{Deserialize, Serialize};

// =============================================================================
// Metadata Types
// =============================================================================

/// A chunk with associated metadata for retrieval and citation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkWithMetadata {
    pub text: String,
    pub metadata: ChunkMetadata,
}

/// Metadata describing a chunk's position and context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub chunk_index: usize,
    pub total_chunks: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub section_header: Option<String>,
    pub parent_chunk_index: Option<usize>,
}

// =============================================================================
// Chunker Trait
// =============================================================================

/// Trait for text chunking implementations
pub trait Chunker: Send + Sync {
    /// Split text into chunks, returning only the text content
    fn chunk(&self, text: &str) -> Vec<String>;

    /// Split text into chunks with metadata (position, section, etc.)
    fn chunk_with_metadata(&self, text: &str) -> Vec<ChunkWithMetadata> {
        let chunks = self.chunk(text);
        let total = chunks.len();
        let mut result = Vec::with_capacity(chunks.len());
        let mut start = 0usize;

        for (i, chunk_text) in chunks.into_iter().enumerate() {
            let end = start + chunk_text.len();
            result.push(ChunkWithMetadata {
                text: chunk_text.clone(),
                metadata: ChunkMetadata {
                    chunk_index: i,
                    total_chunks: total,
                    start_char: start,
                    end_char: end,
                    section_header: None,
                    parent_chunk_index: None,
                },
            });
            start = end;
        }

        result
    }
}

// =============================================================================
// RecursiveCharacterSplitter
// =============================================================================

/// Recursively splits text using a hierarchy of separators
///
/// Inspired by LangChain's RecursiveCharacterTextSplitter. Tries the first
/// separator; if chunks are still too large, recurses with the next separator.
pub struct RecursiveCharacterSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
    separators: Vec<String>,
    strip_whitespace: bool,
}

impl RecursiveCharacterSplitter {
    /// Default separators: paragraph, line, sentence, word, character
    pub fn default_separators() -> Vec<String> {
        vec![
            "\n\n".into(),
            "\n".into(),
            ". ".into(),
            " ".into(),
            "".into(),
        ]
    }

    /// Create a new splitter with chunk size and overlap
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap: chunk_overlap.min(chunk_size.saturating_sub(1)),
            separators: Self::default_separators(),
            strip_whitespace: true,
        }
    }

    /// Set custom separators (order matters: tried first to last)
    pub fn with_separators(mut self, seps: impl Into<Vec<String>>) -> Self {
        self.separators = seps.into();
        self
    }

    /// Set whether to strip whitespace from chunks
    pub fn with_strip_whitespace(mut self, strip: bool) -> Self {
        self.strip_whitespace = strip;
        self
    }

    /// Split text into chunks
    pub fn split(&self, text: &str) -> Vec<String> {
        self.chunk(text)
    }

    fn split_recursive(&self, text: &str, separators: &[String]) -> Vec<String> {
        let text = if self.strip_whitespace {
            text.trim()
        } else {
            text
        };

        if text.is_empty() {
            return vec![];
        }

        let sep = separators
            .first()
            .map(|s| s.as_str())
            .unwrap_or("");

        let (sub_strings, next_seps) = if sep.is_empty() {
            // Character-level split
            (text.chars().map(|c| c.to_string()).collect::<Vec<_>>(), vec![])
        } else {
            let parts: Vec<String> = text.split(sep).map(|s| s.to_string()).collect();
            let next_seps = if separators.len() > 1 {
                separators[1..].to_vec()
            } else {
                vec![]
            };
            (parts, next_seps)
        };

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();

        for (i, part) in sub_strings.iter().enumerate() {
            let part_trimmed = if self.strip_whitespace {
                part.trim()
            } else {
                part
            };

            if part_trimmed.is_empty() {
                continue;
            }

            let to_add = if sep.is_empty() {
                part.clone()
            } else if i > 0 {
                format!("{}{}", sep, part_trimmed)
            } else {
                part_trimmed.to_string()
            };

            if current_chunk.len() + to_add.len() <= self.chunk_size {
                if !current_chunk.is_empty() && !sep.is_empty() {
                    current_chunk.push_str(sep);
                }
                current_chunk.push_str(&to_add);
            } else {
                if !current_chunk.is_empty() {
                    if !next_seps.is_empty() {
                        let mut recursive_chunks =
                            self.split_recursive(&current_chunk, &next_seps);
                        if recursive_chunks.len() > 1 {
                            let last = recursive_chunks.pop().unwrap_or_default();
                            chunks.extend(recursive_chunks);

                            current_chunk = if last.len() > self.chunk_overlap {
                                last.chars()
                                    .skip(last.chars().count().saturating_sub(self.chunk_overlap))
                                    .collect::<String>()
                            } else {
                                last
                            };
                            current_chunk.push_str(&to_add);
                        } else {
                            chunks.push(current_chunk.trim().to_string());
                            current_chunk = to_add;
                        }
                    } else {
                        chunks.push(current_chunk.trim().to_string());
                        current_chunk = to_add;
                    }
                } else {
                    current_chunk = to_add;
                }

                while current_chunk.len() > self.chunk_size {
                    let (chunk, remainder) = self.split_at_size(&current_chunk);
                    let overlap = self.get_overlap_start(&chunk);
                    chunks.push(chunk);
                    current_chunk = overlap;
                    current_chunk.push_str(&remainder);
                }
            }
        }

        if !current_chunk.trim().is_empty() {
            chunks.push(current_chunk.trim().to_string());
        }

        chunks
    }

    fn split_at_size(&self, text: &str) -> (String, String) {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() <= self.chunk_size {
            return (text.to_string(), String::new());
        }
        let split_at = self.chunk_size;
        let chunk: String = chars[..split_at].iter().collect();
        let remainder: String = chars[split_at..].iter().collect();
        (chunk, remainder)
    }

    fn get_overlap_start(&self, text: &str) -> String {
        if self.chunk_overlap == 0 || text.len() <= self.chunk_overlap {
            return String::new();
        }
        let chars: Vec<char> = text.chars().collect();
        let overlap_len = chars.len().saturating_sub(self.chunk_overlap);
        chars[overlap_len..].iter().collect()
    }
}

impl Chunker for RecursiveCharacterSplitter {
    fn chunk(&self, text: &str) -> Vec<String> {
        self.split_recursive(text, &self.separators)
    }
}

// =============================================================================
// MarkdownSplitter
// =============================================================================

/// Splits markdown documents by headers, preserving document structure
pub struct MarkdownSplitter {
    max_chunk_size: usize,
    chunk_overlap: usize,
    include_header_in_chunks: bool,
}

impl MarkdownSplitter {
    /// Create a new markdown splitter
    pub fn new(max_chunk_size: usize) -> Self {
        Self {
            max_chunk_size,
            chunk_overlap: 0,
            include_header_in_chunks: true,
        }
    }

    /// Set overlap between chunks
    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }

    /// Set whether to include header hierarchy in each chunk
    pub fn with_include_headers(mut self, include: bool) -> Self {
        self.include_header_in_chunks = include;
        self
    }

    /// Split markdown into chunks with metadata
    pub fn split(&self, text: &str) -> Vec<ChunkWithMetadata> {
        self.chunk_with_metadata(text)
    }
}

impl Chunker for MarkdownSplitter {
    fn chunk(&self, text: &str) -> Vec<String> {
        self.chunk_with_metadata(text)
            .into_iter()
            .map(|c| c.text)
            .collect()
    }

    fn chunk_with_metadata(&self, text: &str) -> Vec<ChunkWithMetadata> {
        let mut chunks = Vec::new();
        let mut current_header_hierarchy: Vec<String> = Vec::new();
        let mut current_content = String::new();
        let mut current_start = 0usize;
        let mut chunk_index = 0usize;

        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            let line = lines[i];
            let line_start = text[..text.find(line).unwrap_or(0)].len();

            if let Some(header) = parse_markdown_header(line) {
                if !current_content.trim().is_empty() {
                    let content = if self.include_header_in_chunks
                        && !current_header_hierarchy.is_empty()
                    {
                        let prefix = current_header_hierarchy.join(" > ");
                        format!("{}\n\n{}", prefix, current_content.trim())
                    } else {
                        current_content.trim().to_string()
                    };

                    if content.len() > self.max_chunk_size {
                        let fallback = RecursiveCharacterSplitter::new(
                            self.max_chunk_size,
                            self.chunk_overlap,
                        );
                        let sub_chunks = fallback.chunk(&content);
                        for (j, sub) in sub_chunks.into_iter().enumerate() {
                            chunks.push(ChunkWithMetadata {
                                text: sub,
                                metadata: ChunkMetadata {
                                    chunk_index: chunk_index + j,
                                    total_chunks: 0, // Updated at end
                                    start_char: current_start,
                                    end_char: current_start + content.len(),
                                    section_header: Some(current_header_hierarchy.join(" > ")),
                                    parent_chunk_index: if j > 0 {
                                        Some(chunk_index + j - 1)
                                    } else {
                                        None
                                    },
                                },
                            });
                        }
                        chunk_index += chunks.len() - chunk_index;
                    } else {
                        chunks.push(ChunkWithMetadata {
                            text: content.clone(),
                            metadata: ChunkMetadata {
                                chunk_index,
                                total_chunks: 0,
                                start_char: current_start,
                                end_char: current_start + content.len(),
                                section_header: if current_header_hierarchy.is_empty() {
                                    None
                                } else {
                                    Some(current_header_hierarchy.join(" > "))
                                },
                                parent_chunk_index: None,
                            },
                        });
                        chunk_index += 1;
                    }
                    current_start = line_start;
                }

                let (level, title) = header;
                current_header_hierarchy.truncate(level.saturating_sub(1));
                current_header_hierarchy.push(title.to_string());
                current_content = String::new();
            } else {
                if !current_content.is_empty() {
                    current_content.push('\n');
                }
                current_content.push_str(line);
            }
            i += 1;
        }

        if !current_content.trim().is_empty() {
            let content = if self.include_header_in_chunks && !current_header_hierarchy.is_empty() {
                let prefix = current_header_hierarchy.join(" > ");
                format!("{}\n\n{}", prefix, current_content.trim())
            } else {
                current_content.trim().to_string()
            };

            if content.len() > self.max_chunk_size {
                let fallback = RecursiveCharacterSplitter::new(
                    self.max_chunk_size,
                    self.chunk_overlap,
                );
                let sub_chunks = fallback.chunk(&content);
                for (j, sub) in sub_chunks.into_iter().enumerate() {
                    chunks.push(ChunkWithMetadata {
                        text: sub,
                        metadata: ChunkMetadata {
                            chunk_index: chunk_index + j,
                            total_chunks: 0,
                            start_char: current_start,
                            end_char: current_start + content.len(),
                            section_header: Some(current_header_hierarchy.join(" > ")),
                            parent_chunk_index: if j > 0 {
                                Some(chunk_index + j - 1)
                            } else {
                                None
                            },
                        },
                    });
                }
            } else {
                chunks.push(ChunkWithMetadata {
                    text: content.clone(),
                    metadata: ChunkMetadata {
                        chunk_index,
                        total_chunks: 0,
                        start_char: current_start,
                        end_char: current_start + content.len(),
                        section_header: if current_header_hierarchy.is_empty() {
                            None
                        } else {
                            Some(current_header_hierarchy.join(" > "))
                        },
                        parent_chunk_index: None,
                    },
                });
            }
        }

        let total = chunks.len();
        for c in &mut chunks {
            c.metadata.total_chunks = total;
        }

        chunks
    }
}

fn parse_markdown_header(line: &str) -> Option<(usize, &str)> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut level = 0usize;
    for c in trimmed.chars() {
        if c == '#' {
            level += 1;
        } else {
            break;
        }
    }
    if level > 0 && level <= 6 {
        let title = trimmed[level..].trim();
        if !title.is_empty() {
            return Some((level, title));
        }
    }
    None
}

// =============================================================================
// SlidingWindowSplitter
// =============================================================================

/// Sliding window chunker with exact character-level overlap control
pub struct SlidingWindowSplitter {
    window_size: usize,
    step_size: usize,
    respect_word_boundaries: bool,
    respect_sentence_boundaries: bool,
}

impl SlidingWindowSplitter {
    /// Create a new sliding window splitter
    /// `overlap` is the number of characters that overlap between consecutive windows
    pub fn new(window_size: usize, overlap: usize) -> Self {
        let step_size = window_size.saturating_sub(overlap).max(1);
        Self {
            window_size,
            step_size,
            respect_word_boundaries: false,
            respect_sentence_boundaries: false,
        }
    }

    /// Respect word boundaries when splitting
    pub fn with_word_boundaries(mut self, respect: bool) -> Self {
        self.respect_word_boundaries = respect;
        self
    }

    /// Respect sentence boundaries when splitting
    pub fn with_sentence_boundaries(mut self, respect: bool) -> Self {
        self.respect_sentence_boundaries = respect;
        self
    }

    /// Split text using sliding window
    pub fn split(&self, text: &str) -> Vec<String> {
        self.chunk(text)
    }
}

impl Chunker for SlidingWindowSplitter {
    fn chunk(&self, text: &str) -> Vec<String> {
        let text = text.trim();
        if text.is_empty() {
            return vec![];
        }

        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();

        let mut start = 0usize;
        while start < len {
            let mut end = (start + self.window_size).min(len);

            if self.respect_word_boundaries || self.respect_sentence_boundaries {
                if end < len {
                    let chunk: String = chars[start..end].iter().collect();
                    let mut new_end = end;

                    if self.respect_sentence_boundaries {
                        if let Some(idx) = chunk.rfind(|c| c == '.' || c == '!' || c == '?') {
                            new_end = start + idx + 1;
                        }
                    }
                    if self.respect_word_boundaries && new_end == end {
                        if let Some(idx) = chunk.rfind(' ') {
                            new_end = start + idx + 1;
                        }
                    }
                    end = new_end;
                }
            }

            let chunk: String = chars[start..end].iter().collect();
            let trimmed = chunk.trim();
            if !trimmed.is_empty() {
                chunks.push(trimmed.to_string());
            }

            start += self.step_size;
            if start >= len {
                break;
            }
            if start + self.step_size > len && start < len {
                start = len.saturating_sub(self.window_size);
                if start <= chunks.len().saturating_sub(1) * self.step_size {
                    break;
                }
            }
        }

        chunks
    }
}

// =============================================================================
// SentenceSplitter
// =============================================================================

/// Groups sentences into chunks up to a size limit
pub struct SentenceSplitter {
    max_chunk_size: usize,
    overlap_sentences: usize,
}

impl SentenceSplitter {
    /// Create a new sentence splitter
    pub fn new(max_chunk_size: usize) -> Self {
        Self {
            max_chunk_size,
            overlap_sentences: 0,
        }
    }

    /// Set overlap in number of sentences
    pub fn with_overlap(mut self, n_sentences: usize) -> Self {
        self.overlap_sentences = n_sentences;
        self
    }

    /// Split text into chunks
    pub fn split(&self, text: &str) -> Vec<String> {
        self.chunk(text)
    }
}

impl Chunker for SentenceSplitter {
    fn chunk(&self, text: &str) -> Vec<String> {
        let sentences = split_sentences(text);
        if sentences.is_empty() {
            return if text.trim().is_empty() {
                vec![]
            } else {
                vec![text.trim().to_string()]
            };
        }

        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut i = 0;

        while i < sentences.len() {
            let sent = &sentences[i];
            let to_add = if sent.ends_with('.') || sent.ends_with('!') || sent.ends_with('?') {
                sent.clone()
            } else {
                format!("{}.", sent)
            };

            if current.len() + to_add.len() + 1 <= self.max_chunk_size || current.is_empty() {
                if !current.is_empty() {
                    current.push(' ');
                }
                current.push_str(&to_add);
                i += 1;
            } else {
                if !current.is_empty() {
                    chunks.push(current.trim().to_string());
                    let overlap_count = self.overlap_sentences.min(chunks.len() * 2);
                    let overlap_sents: Vec<_> = sentences[i.saturating_sub(overlap_count)..i]
                        .iter()
                        .rev()
                        .take(self.overlap_sentences)
                        .cloned()
                        .collect();
                    current = overlap_sents.into_iter().rev().collect::<Vec<_>>().join(" ");
                }
            }
        }

        if !current.trim().is_empty() {
            chunks.push(current.trim().to_string());
        }

        chunks
    }
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        current.push(c);
        if c == '.' || c == '!' || c == '?' || (c == '\n' && current.ends_with("\n\n")) {
            let s = current.trim().to_string();
            if !s.is_empty() {
                sentences.push(s);
            }
            current.clear();
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }
    sentences
}

// =============================================================================
// ParagraphSplitter
// =============================================================================

/// Splits text on paragraph boundaries (double newlines)
pub struct ParagraphSplitter {
    max_chunk_size: usize,
    merge_short_paragraphs: bool,
    min_paragraph_size: usize,
}

impl ParagraphSplitter {
    /// Create a new paragraph splitter
    pub fn new(max_chunk_size: usize) -> Self {
        Self {
            max_chunk_size,
            merge_short_paragraphs: false,
            min_paragraph_size: 0,
        }
    }

    /// Merge short paragraphs together until min_size
    pub fn with_merge(mut self, min_size: usize) -> Self {
        self.merge_short_paragraphs = true;
        self.min_paragraph_size = min_size;
        self
    }

    /// Split text into chunks
    pub fn split(&self, text: &str) -> Vec<String> {
        self.chunk(text)
    }
}

impl Chunker for ParagraphSplitter {
    fn chunk(&self, text: &str) -> Vec<String> {
        let paragraphs: Vec<&str> = text.split("\n\n").filter(|p| !p.trim().is_empty()).collect();

        if paragraphs.is_empty() {
            return if text.trim().is_empty() {
                vec![]
            } else {
                vec![text.trim().to_string()]
            };
        }

        let mut chunks = Vec::new();
        let mut current = String::new();

        for para in paragraphs {
            let para = para.trim();
            if para.is_empty() {
                continue;
            }

            let to_add = if current.is_empty() {
                para.to_string()
            } else {
                format!("\n\n{}", para)
            };

            if self.merge_short_paragraphs
                && current.len() < self.min_paragraph_size
                && current.len() + to_add.len() <= self.max_chunk_size
            {
                current.push_str(&to_add);
            } else if current.len() + to_add.len() <= self.max_chunk_size {
                current.push_str(&to_add);
            } else {
                if !current.is_empty() {
                    chunks.push(current.trim().to_string());
                }
                current = para.to_string();
            }
        }

        if !current.trim().is_empty() {
            chunks.push(current.trim().to_string());
        }

        chunks
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recursive_splitter_basic() {
        let splitter = RecursiveCharacterSplitter::new(100, 20);
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph with more content.";
        let chunks = splitter.split(text);
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| !c.is_empty()));
    }

    #[test]
    fn test_recursive_splitter_respects_size() {
        let splitter = RecursiveCharacterSplitter::new(50, 10);
        let text = "A".repeat(200);
        let chunks = splitter.split(&text);
        for chunk in &chunks {
            assert!(
                chunk.len() <= 60,
                "chunk len {} exceeds 60: {:?}",
                chunk.len(),
                chunk
            );
        }
    }

    #[test]
    fn test_recursive_splitter_overlap() {
        let splitter = RecursiveCharacterSplitter::new(30, 10);
        let text = "one two three four five six seven eight nine ten";
        let chunks = splitter.split(text);
        if chunks.len() >= 2 {
            let c1 = &chunks[0];
            let c2 = &chunks[1];
            let overlap = c1.chars().rev().take(10).collect::<String>();
            let overlap_rev: String = overlap.chars().rev().collect();
            assert!(
                c2.starts_with(&overlap_rev) || c1.ends_with(&c2[..c2.len().min(10)]),
                "chunks should have overlap"
            );
        }
    }

    #[test]
    fn test_markdown_splitter_headers() {
        let splitter = MarkdownSplitter::new(500);
        let text = "# Title\n\nContent one.\n\n## Section\n\nContent two.";
        let chunks = splitter.split(text);
        assert!(chunks.len() >= 2);
        assert!(chunks[0].text.contains("Content one") || chunks[0].text.contains("Title"));
        assert!(chunks.iter().any(|c| c.text.contains("Content two")));
    }

    #[test]
    fn test_markdown_splitter_includes_hierarchy() {
        let splitter = MarkdownSplitter::new(500).with_include_headers(true);
        let text = "# Main\n\nIntro.\n\n## Sub\n\nDetails.";
        let chunks = splitter.chunk_with_metadata(text);
        assert!(!chunks.is_empty());
        let with_header = chunks
            .iter()
            .find(|c| c.metadata.section_header.is_some())
            .unwrap();
        assert!(with_header.metadata.section_header.as_ref().unwrap().contains("Main"));
    }

    #[test]
    fn test_sliding_window_exact_overlap() {
        let splitter = SlidingWindowSplitter::new(20, 5);
        let text = "abcdefghijklmnopqrstuvwxyz";
        let chunks = splitter.split(text);
        assert!(!chunks.is_empty());
        assert_eq!(splitter.step_size, 15);
    }

    #[test]
    fn test_sliding_window_word_boundaries() {
        let splitter =
            SlidingWindowSplitter::new(25, 5).with_word_boundaries(true);
        let text = "hello world foo bar baz qux";
        let chunks = splitter.split(text);
        for chunk in &chunks {
            if chunk.len() < 25 {
                assert!(
                    chunk.chars().last().map(|c| c != ' ').unwrap_or(true),
                    "chunk should not end mid-word"
                );
            }
        }
    }

    #[test]
    fn test_sentence_splitter() {
        let splitter = SentenceSplitter::new(50).with_overlap(1);
        let text = "First sentence. Second sentence. Third sentence. Fourth.";
        let chunks = splitter.split(text);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.contains('.'));
        }
    }

    #[test]
    fn test_paragraph_splitter() {
        let splitter = ParagraphSplitter::new(12);
        let text = "Para one.\n\nPara two.\n\nPara three.";
        let chunks = splitter.split(text);
        assert_eq!(chunks.len(), 3, "expected 3 chunks, got {:?}", chunks);
        assert!(chunks[0].contains("Para one"));
        assert!(chunks[1].contains("Para two"));
        assert!(chunks[2].contains("Para three"));
    }

    #[test]
    fn test_paragraph_merge_short() {
        let splitter = ParagraphSplitter::new(100).with_merge(50);
        let text = "Short.\n\nAlso short.\n\nAnother.";
        let chunks = splitter.split(text);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_chunk_metadata() {
        let splitter = RecursiveCharacterSplitter::new(20, 5);
        let text = "Hello world. This is a test.";
        let chunks = splitter.chunk_with_metadata(text);
        assert!(!chunks.is_empty());
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.metadata.chunk_index, i);
            assert_eq!(c.metadata.total_chunks, chunks.len());
            assert!(c.metadata.start_char <= c.metadata.end_char);
        }
    }

    #[test]
    fn test_chunker_trait_impl() {
        let splitter = RecursiveCharacterSplitter::new(50, 10);
        let text = "Some text to chunk.";
        let simple = splitter.chunk(text);
        let with_meta = splitter.chunk_with_metadata(text);
        assert_eq!(simple.len(), with_meta.len());
        for (a, b) in simple.iter().zip(with_meta.iter()) {
            assert_eq!(a, &b.text);
        }
    }
}
