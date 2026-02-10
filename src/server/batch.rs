//! Continuous batching for efficient multi-request processing
//!
//! This module provides a batch scheduler that manages multiple concurrent
//! generation requests, batching tokens together for efficient processing.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{Mutex, mpsc};

/// Request ID for tracking
pub type RequestId = u64;

/// Generation request
#[derive(Debug)]
pub struct GenerationRequest {
    /// Unique request ID
    pub id: RequestId,
    /// Input token IDs
    pub input_tokens: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-P for sampling
    pub top_p: f32,
    /// Stop sequences (token IDs)
    pub stop_sequences: Vec<Vec<u32>>,
    /// Channel to send generated tokens
    pub token_sender: mpsc::Sender<GenerationEvent>,
}

/// Events sent during generation
#[derive(Debug, Clone)]
pub enum GenerationEvent {
    /// A new token was generated
    Token(u32),
    /// Generation completed
    Finished { reason: FinishReason },
    /// Error occurred
    Error(String),
}

/// Reason for generation finishing
#[derive(Debug, Clone)]
pub enum FinishReason {
    /// Reached max tokens
    MaxTokens,
    /// Hit end of sequence token
    EndOfSequence,
    /// Hit stop sequence
    StopSequence,
    /// Cancelled by user
    Cancelled,
}

/// Sequence state in the batch
#[derive(Debug)]
struct Sequence {
    /// Request ID
    request_id: RequestId,
    /// Current tokens (prompt + generated)
    tokens: Vec<u32>,
    /// Number of prompt tokens
    prompt_len: usize,
    /// Number of tokens generated so far
    generated: usize,
    /// Maximum tokens to generate
    max_tokens: usize,
    /// Temperature
    temperature: f32,
    /// Top-P
    top_p: f32,
    /// Stop sequences
    stop_sequences: Vec<Vec<u32>>,
    /// Channel to send events
    token_sender: mpsc::Sender<GenerationEvent>,
    /// Current position in KV cache
    cache_position: usize,
}

/// Batch scheduler configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size (concurrent sequences)
    pub max_batch_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Timeout for waiting on batch (milliseconds)
    pub batch_timeout_ms: u64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_seq_len: 4096,
            batch_timeout_ms: 10,
        }
    }
}

/// Batch scheduler for continuous batching
pub struct BatchScheduler {
    /// Configuration
    config: BatchConfig,
    /// Active sequences
    sequences: HashMap<RequestId, Sequence>,
    /// Next request ID
    next_id: RequestId,
    /// Pending requests waiting for a slot
    pending: Vec<GenerationRequest>,
}

impl BatchScheduler {
    /// Create a new batch scheduler
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            sequences: HashMap::new(),
            next_id: 1,
            pending: Vec::new(),
        }
    }

    /// Add a new generation request
    ///
    /// Returns the request ID and a channel to receive results
    pub fn add_request(&mut self, mut request: GenerationRequest) -> RequestId {
        let id = self.next_id;
        self.next_id += 1;
        request.id = id;

        if self.sequences.len() < self.config.max_batch_size {
            // Add immediately
            self.add_to_batch(request);
        } else {
            // Queue for later
            self.pending.push(request);
        }

        id
    }

    /// Add a request to the active batch
    fn add_to_batch(&mut self, request: GenerationRequest) {
        let seq = Sequence {
            request_id: request.id,
            tokens: request.input_tokens.clone(),
            prompt_len: request.input_tokens.len(),
            generated: 0,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stop_sequences: request.stop_sequences,
            token_sender: request.token_sender,
            cache_position: 0,
        };

        self.sequences.insert(request.id, seq);
    }

    /// Cancel a request
    pub fn cancel_request(&mut self, id: RequestId) {
        if let Some(seq) = self.sequences.remove(&id) {
            let _ = seq.token_sender.try_send(GenerationEvent::Finished {
                reason: FinishReason::Cancelled,
            });
        }

        // Also check pending
        self.pending.retain(|r| r.id != id);

        // Promote pending request if there's now a slot
        self.promote_pending();
    }

    /// Promote a pending request to active
    fn promote_pending(&mut self) {
        while self.sequences.len() < self.config.max_batch_size && !self.pending.is_empty() {
            let request = self.pending.remove(0);
            self.add_to_batch(request);
        }
    }

    /// Get the current batch of sequences for processing
    pub fn get_batch(&self) -> Vec<RequestId> {
        self.sequences.keys().copied().collect()
    }

    /// Get the tokens to process for a sequence
    pub fn get_sequence_tokens(&self, id: RequestId) -> Option<&[u32]> {
        self.sequences.get(&id).map(|s| s.tokens.as_slice())
    }

    /// Get sequence info
    pub fn get_sequence_info(&self, id: RequestId) -> Option<(f32, f32)> {
        self.sequences.get(&id).map(|s| (s.temperature, s.top_p))
    }

    /// Process generated token for a sequence
    ///
    /// Returns true if sequence should continue, false if finished
    pub fn process_token(&mut self, id: RequestId, token: u32, eos_token: u32) -> bool {
        let (should_stop, finish_reason) = {
            let seq = match self.sequences.get_mut(&id) {
                Some(s) => s,
                None => return false,
            };

            // Add token to sequence
            seq.tokens.push(token);
            seq.generated += 1;

            // Send token to client
            let _ = seq.token_sender.try_send(GenerationEvent::Token(token));

            // Check termination conditions
            if token == eos_token {
                (true, Some(FinishReason::EndOfSequence))
            } else if seq.generated >= seq.max_tokens {
                (true, Some(FinishReason::MaxTokens))
            } else if Self::check_stop_sequence_static(&seq.tokens, &seq.stop_sequences) {
                (true, Some(FinishReason::StopSequence))
            } else {
                (false, None)
            }
        };

        if should_stop {
            if let Some(seq) = self.sequences.remove(&id)
                && let Some(reason) = finish_reason
            {
                let _ = seq
                    .token_sender
                    .try_send(GenerationEvent::Finished { reason });
            }
            self.promote_pending();
        }

        !should_stop
    }

    /// Check if a stop sequence was hit (static version for borrow checker)
    fn check_stop_sequence_static(tokens: &[u32], stop_sequences: &[Vec<u32>]) -> bool {
        for stop_seq in stop_sequences {
            if tokens.len() >= stop_seq.len() {
                let end = &tokens[tokens.len() - stop_seq.len()..];
                if end == stop_seq {
                    return true;
                }
            }
        }
        false
    }

    /// Report an error for a sequence
    pub fn report_error(&mut self, id: RequestId, error: String) {
        if let Some(seq) = self.sequences.remove(&id) {
            let _ = seq.token_sender.try_send(GenerationEvent::Error(error));
        }
        self.promote_pending();
    }

    /// Get number of active sequences
    pub fn active_count(&self) -> usize {
        self.sequences.len()
    }

    /// Get number of pending requests
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Check if scheduler has work
    pub fn has_work(&self) -> bool {
        !self.sequences.is_empty()
    }
}

/// Thread-safe batch scheduler wrapper
pub type SharedBatchScheduler = Arc<Mutex<BatchScheduler>>;

/// Create a new shared batch scheduler
pub fn new_batch_scheduler(config: BatchConfig) -> SharedBatchScheduler {
    Arc::new(Mutex::new(BatchScheduler::new(config)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_batch_scheduler_basic() {
        let scheduler = BatchScheduler::new(BatchConfig::default());
        assert_eq!(scheduler.active_count(), 0);
        assert_eq!(scheduler.pending_count(), 0);
    }

    #[tokio::test]
    async fn test_batch_scheduler_add_request() {
        let mut scheduler = BatchScheduler::new(BatchConfig {
            max_batch_size: 2,
            ..Default::default()
        });

        let (tx, _rx) = mpsc::channel(100);

        let request = GenerationRequest {
            id: 0,
            input_tokens: vec![1, 2, 3],
            max_tokens: 10,
            temperature: 0.8,
            top_p: 0.9,
            stop_sequences: vec![],
            token_sender: tx,
        };

        let id = scheduler.add_request(request);
        assert_eq!(id, 1);
        assert_eq!(scheduler.active_count(), 1);
    }

    #[tokio::test]
    async fn test_batch_scheduler_overflow() {
        let mut scheduler = BatchScheduler::new(BatchConfig {
            max_batch_size: 1,
            ..Default::default()
        });

        for _ in 0..3 {
            let (tx, _rx) = mpsc::channel(100);
            let request = GenerationRequest {
                id: 0,
                input_tokens: vec![1, 2, 3],
                max_tokens: 10,
                temperature: 0.8,
                top_p: 0.9,
                stop_sequences: vec![],
                token_sender: tx,
            };
            scheduler.add_request(request);
        }

        assert_eq!(scheduler.active_count(), 1);
        assert_eq!(scheduler.pending_count(), 2);
    }
}
