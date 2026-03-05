//! Batched inference engine for continuous batching
//!
//! Provides `BatchedEngine` that processes multiple generation requests
//! concurrently using continuous batching. Each iteration processes one
//! token per active sequence, maximizing GPU utilization.

#![cfg(feature = "server")]

use std::sync::Arc;

use tokio::sync::{mpsc, Mutex};

use crate::backend::Backend;
use crate::model::{InferenceContext, Model, ModelConfig};
use crate::sampling::{Sampler, SamplerConfig};
use crate::tokenizer::Tokenizer;

// ============================================================================
// Config
// ============================================================================

#[derive(Debug, Clone)]
pub struct BatchedEngineConfig {
    /// Maximum concurrent sequences
    pub max_batch_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Maximum queued requests (beyond this, reject)
    pub max_queue_depth: usize,
}

impl Default for BatchedEngineConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_seq_len: 4096,
            max_queue_depth: 64,
        }
    }
}

// ============================================================================
// Request / Response types
// ============================================================================

/// A request submitted to the batched engine
pub struct BatchRequest {
    /// Prompt tokens
    pub tokens: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Sampler configuration
    pub sampler_config: SamplerConfig,
    /// Channel to receive generated tokens
    pub token_sender: mpsc::Sender<BatchToken>,
}

/// Token event from the batched engine
#[derive(Debug, Clone)]
pub enum BatchToken {
    /// A generated token
    Token { id: u32, text: String },
    /// Generation finished
    Done {
        reason: BatchFinishReason,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    /// Error occurred
    Error(String),
}

/// Reason for finishing
#[derive(Debug, Clone)]
pub enum BatchFinishReason {
    Stop,
    MaxTokens,
    Error,
}

// ============================================================================
// Internal state
// ============================================================================

/// Internal state for an active sequence in the batch
struct ActiveSequence {
    /// All tokens so far (prompt + generated)
    tokens: Vec<u32>,
    /// Prompt length
    prompt_len: usize,
    /// Number of generated tokens
    generated: usize,
    /// Maximum tokens to generate
    max_tokens: usize,
    /// Inference context with KV cache
    ctx: InferenceContext,
    /// Sampler for this sequence
    sampler: Sampler,
    /// Channel to send results
    sender: mpsc::Sender<BatchToken>,
}

/// Command for the background loop
enum BatchCommand {
    Request(BatchRequest),
    Shutdown,
}

// ============================================================================
// BatchedEngine
// ============================================================================

/// Batched inference engine using continuous batching
pub struct BatchedEngine {
    config: BatchedEngineConfig,
    /// Channel to submit new requests
    request_tx: mpsc::Sender<BatchCommand>,
    /// Queue depth counter (active + pending)
    queue_count: Arc<Mutex<usize>>,
    /// Handle to the background processing loop
    _handle: Option<tokio::task::JoinHandle<()>>,
}

impl BatchedEngine {
    /// Create a new batched engine and spawn the background processing loop.
    pub fn new(
        model: Arc<dyn Model>,
        tokenizer: Arc<Tokenizer>,
        _model_config: ModelConfig,
        backend: Arc<dyn Backend>,
        config: BatchedEngineConfig,
    ) -> Self {
        let (request_tx, mut request_rx) = mpsc::channel(config.max_queue_depth);
        let queue_count = Arc::new(Mutex::new(0));

        let model_clone = model.clone();
        let tokenizer_clone = tokenizer.clone();
        let backend_clone = backend.clone();
        let queue_count_clone = queue_count.clone();
        let max_batch_size = config.max_batch_size;
        let max_seq_len = config.max_seq_len;
        let eos_token_id = tokenizer.special_tokens.eos_token_id;

        let handle = tokio::spawn(async move {
            run_background_loop(
                model_clone,
                tokenizer_clone,
                backend_clone,
                &mut request_rx,
                queue_count_clone,
                max_batch_size,
                max_seq_len,
                eos_token_id,
            )
            .await;
        });

        Self {
            config,
            request_tx,
            queue_count,
            _handle: Some(handle),
        }
    }

    /// Submit a request. Returns error if queue is full.
    pub fn submit(&self, request: BatchRequest) -> Result<(), String> {
        let mut count = self
            .queue_count
            .try_lock()
            .map_err(|_| "failed to lock queue")?;

        if *count >= self.config.max_queue_depth {
            return Err("queue full".to_string());
        }

        *count += 1;
        drop(count);

        self.request_tx
            .try_send(BatchCommand::Request(request))
            .map_err(|e| {
                // Decrement on send failure
                if let Ok(mut c) = self.queue_count.try_lock() {
                    *c = c.saturating_sub(1);
                }
                e.to_string()
            })?;

        Ok(())
    }

    /// Signal the background loop to stop.
    pub fn shutdown(&self) {
        let _ = self.request_tx.try_send(BatchCommand::Shutdown);
    }
}

/// Background loop: receive requests, process active sequences, send results.
async fn run_background_loop(
    model: Arc<dyn Model>,
    tokenizer: Arc<Tokenizer>,
    backend: Arc<dyn Backend>,
    request_rx: &mut mpsc::Receiver<BatchCommand>,
    queue_count: Arc<Mutex<usize>>,
    max_batch_size: usize,
    max_seq_len: usize,
    eos_token_id: u32,
) {
    let mut active: Vec<ActiveSequence> = Vec::with_capacity(max_batch_size);
    let mut pending: Vec<BatchRequest> = Vec::new();
    let mut shutdown = false;

    while !shutdown {
        // 1. Drain new requests (non-blocking)
        while let Ok(cmd) = request_rx.try_recv() {
            match cmd {
                BatchCommand::Request(req) => {
                    if active.len() < max_batch_size {
                        if let Some(seq) = create_active_sequence(
                            req,
                            &model,
                            &tokenizer,
                            &backend,
                            max_seq_len,
                        ) {
                            active.push(seq);
                        } else {
                            decrement_queue(&queue_count).await;
                        }
                    } else {
                        pending.push(req);
                    }
                }
                BatchCommand::Shutdown => shutdown = true,
            }
        }

        // 2. Process each active sequence (one token per iteration)
        let mut i = 0;
        while i < active.len() {
            let seq = &mut active[i];
            let result = step_sequence(seq, &model, &tokenizer, eos_token_id);

            match result {
                Ok(Some((token_id, text))) => {
                    let _ = seq
                        .sender
                        .send(BatchToken::Token {
                            id: token_id,
                            text,
                        })
                        .await;
                }
                Ok(None) => {
                    // Sequence finished
                    let prompt_tokens = seq.prompt_len;
                    let completion_tokens = seq.generated;
                    let reason = if seq.generated >= seq.max_tokens {
                        BatchFinishReason::MaxTokens
                    } else {
                        BatchFinishReason::Stop
                    };
                    let sender = seq.sender.clone();
                    active.remove(i);
                    decrement_queue(&queue_count).await;
                    let _ = sender
                        .send(BatchToken::Done {
                            reason,
                            prompt_tokens,
                            completion_tokens,
                        })
                        .await;
                    continue;
                }
                Err(e) => {
                    let sender = seq.sender.clone();
                    active.remove(i);
                    decrement_queue(&queue_count).await;
                    let _ = sender
                        .send(BatchToken::Error(e.to_string()))
                        .await;
                    continue;
                }
            }
            i += 1;
        }

        // 3. Promote pending to active when we have space
        while active.len() < max_batch_size {
            match pending.pop() {
                Some(req) => {
                    if let Some(seq) =
                        create_active_sequence(req, &model, &tokenizer, &backend, max_seq_len)
                    {
                        active.push(seq);
                    } else {
                        decrement_queue(&queue_count).await;
                    }
                }
                None => break,
            }
        }

        if shutdown {
            break;
        }

        // 4. Sleep briefly if no work
        if active.is_empty() {
            match tokio::time::timeout(
                std::time::Duration::from_millis(10),
                request_rx.recv(),
            )
            .await
            {
                Ok(Some(BatchCommand::Request(req))) => {
                    if let Some(seq) =
                        create_active_sequence(req, &model, &tokenizer, &backend, max_seq_len)
                    {
                        active.push(seq);
                    } else {
                        decrement_queue(&queue_count).await;
                    }
                }
                Ok(Some(BatchCommand::Shutdown)) => break,
                Ok(None) => break,
                Err(_) => {}
            }
        }
    }
}

async fn decrement_queue(queue_count: &Arc<Mutex<usize>>) {
    let mut c = queue_count.lock().await;
    *c = c.saturating_sub(1);
}

fn create_active_sequence(
    req: BatchRequest,
    model: &Arc<dyn Model>,
    _tokenizer: &Arc<Tokenizer>,
    backend: &Arc<dyn Backend>,
    max_seq_len: usize,
) -> Option<ActiveSequence> {
    if req.tokens.is_empty() {
        let _ = req.token_sender.try_send(BatchToken::Error(
            "empty prompt".to_string(),
        ));
        return None;
    }

    let prompt_len = req.tokens.len().min(max_seq_len.saturating_sub(1));
    let tokens: Vec<u32> = req.tokens.iter().take(prompt_len).copied().collect();
    let prompt_len = tokens.len();

    let ctx = model.create_context(backend.clone());
    let sampler = Sampler::new(req.sampler_config.clone(), model.vocab_size());

    Some(ActiveSequence {
        tokens: tokens.clone(),
        prompt_len,
        generated: 0,
        max_tokens: req.max_tokens,
        ctx,
        sampler,
        sender: req.token_sender,
    })
}

/// Step one sequence: prefill or decode one token. Returns Ok(Some((id, text))),
/// Ok(None) if done, or Err on model error.
fn step_sequence(
    seq: &mut ActiveSequence,
    model: &Arc<dyn Model>,
    tokenizer: &Arc<Tokenizer>,
    eos_token_id: u32,
) -> Result<Option<(u32, String)>, crate::model::ModelError> {
    // Check EOS from last token
    if let Some(&last) = seq.tokens.last() {
        if last == eos_token_id {
            return Ok(None);
        }
    }

    if seq.generated >= seq.max_tokens {
        return Ok(None);
    }

    let input_tokens: &[u32] = if seq.ctx.position == 0 {
        &seq.tokens[..]
    } else {
        &seq.tokens[seq.tokens.len().saturating_sub(1)..]
    };

    let logits = model.forward(input_tokens, &mut seq.ctx)?;
    let next_token = seq.sampler.sample(&logits, &seq.tokens);

    seq.tokens.push(next_token);
    seq.generated += 1;

    if next_token == eos_token_id {
        return Ok(None);
    }

    let text = tokenizer
        .decode_token(next_token)
        .unwrap_or_else(|_| String::new());

    Ok(Some((next_token, text)))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batched_engine_config_default() {
        let config = BatchedEngineConfig::default();
        assert_eq!(config.max_batch_size, 8);
        assert_eq!(config.max_seq_len, 4096);
        assert_eq!(config.max_queue_depth, 64);
    }

    #[test]
    fn test_batch_request_creation() {
        let (tx, _rx) = mpsc::channel(1);
        let req = BatchRequest {
            tokens: vec![1, 2, 3],
            max_tokens: 64,
            sampler_config: SamplerConfig::default(),
            token_sender: tx,
        };
        assert_eq!(req.tokens.len(), 3);
        assert_eq!(req.max_tokens, 64);
    }

    #[test]
    fn test_batch_finish_reason() {
        let stop = BatchFinishReason::Stop;
        let max = BatchFinishReason::MaxTokens;
        let err = BatchFinishReason::Error;

        match &stop {
            BatchFinishReason::Stop => {}
            _ => panic!("expected Stop"),
        }
        match &max {
            BatchFinishReason::MaxTokens => {}
            _ => panic!("expected MaxTokens"),
        }
        match &err {
            BatchFinishReason::Error => {}
            _ => panic!("expected Error"),
        }
    }
}
