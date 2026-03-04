//! OpenAI API compatible types

use serde::{Deserialize, Serialize};

/// Chat message role
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(default)]
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Tool definition for function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

/// Function definition within a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Tool call in assistant response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

/// Function call within a tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Tool choice for function calling
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    Specific { #[serde(rename = "type")] tool_type: String, function: ToolChoiceFunction },
}

/// Specific function to call
#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

/// Chat completion request
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
}

fn default_max_tokens() -> usize {
    256
}

fn default_temperature() -> f32 {
    0.7
}

fn default_top_p() -> f32 {
    0.9
}

/// Chat completion choice
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Chat completion response
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

/// Usage statistics
#[derive(Debug, Clone, Serialize, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Streaming chat completion chunk
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Streaming choice delta
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunkChoice {
    pub index: usize,
    pub delta: ChatCompletionDelta,
    pub finish_reason: Option<String>,
}

/// Content delta for streaming
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

// =============================================================================
// Embeddings API Types
// =============================================================================

/// Embedding request (OpenAI-compatible)
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingRequest {
    pub input: EmbeddingInput,
    #[serde(default)]
    pub model: String,
    #[serde(default = "default_encoding_format")]
    pub encoding_format: String,
}

/// Embedding input: single string or array of strings
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

fn default_encoding_format() -> String {
    "float".to_string()
}

/// Embedding response
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// Single embedding result
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

/// Embedding usage statistics
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

// =============================================================================
// Model hot-swap types
// =============================================================================

/// Request to load/swap a model
#[derive(Debug, Clone, Deserialize)]
pub struct LoadModelRequest {
    pub model_path: String,
}

/// Response after model load
#[derive(Debug, Clone, Serialize)]
pub struct LoadModelResponse {
    pub status: String,
    pub model: String,
    pub context_size: usize,
}

/// Server queue status
#[derive(Debug, Clone, Serialize)]
pub struct QueueStatusResponse {
    pub active_requests: usize,
    pub queued_requests: usize,
    pub max_queue_depth: usize,
    pub max_concurrent: usize,
}

/// Text completion request
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    /// Model to use
    #[serde(default)]
    pub model: String,
    /// Prompt text
    pub prompt: String,
    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Sampling temperature
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Top-P sampling
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Whether to stream
    #[serde(default)]
    pub stream: bool,
    /// Stop sequences
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

/// Text completion choice
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: usize,
    pub finish_reason: String,
}

/// Text completion response
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

/// Model info for /v1/models endpoint
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// Models list response
#[derive(Debug, Clone, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Health check response
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub context_size: usize,
}

/// Error response
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}

impl ErrorResponse {
    pub fn new(message: impl Into<String>, error_type: impl Into<String>) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                r#type: error_type.into(),
                code: None,
            },
        }
    }
}

// =============================================================================
// RAG / Knowledge Base API Types (Bedrock-style)
// =============================================================================

/// Request to retrieve from a knowledge base
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrieveRequest {
    /// Knowledge base ID/name
    pub knowledge_base_id: String,
    /// Query text
    pub query: String,
    /// Retrieval configuration
    #[serde(default)]
    pub retrieval_configuration: Option<RetrievalConfiguration>,
    /// Token for pagination
    #[serde(default)]
    pub next_token: Option<String>,
}

/// Retrieval configuration
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalConfiguration {
    /// Vector search configuration
    #[serde(default)]
    pub vector_search_configuration: Option<VectorSearchConfiguration>,
}

/// Vector search configuration
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VectorSearchConfiguration {
    /// Number of results to return
    #[serde(default = "default_num_results")]
    pub number_of_results: usize,
    /// Override search type
    #[serde(default)]
    pub override_search_type: Option<String>,
    /// Metadata filter
    #[serde(default)]
    pub filter: Option<RetrievalFilter>,
}

fn default_num_results() -> usize {
    5
}

/// Retrieval filter
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalFilter {
    /// AND conditions
    #[serde(default)]
    pub and_all: Option<Vec<RetrievalFilter>>,
    /// OR conditions
    #[serde(default)]
    pub or_all: Option<Vec<RetrievalFilter>>,
    /// Equals condition
    #[serde(default)]
    pub equals: Option<FilterCondition>,
    /// Not equals condition
    #[serde(default)]
    pub not_equals: Option<FilterCondition>,
    /// Greater than condition
    #[serde(default)]
    pub greater_than: Option<FilterCondition>,
    /// Less than condition
    #[serde(default)]
    pub less_than: Option<FilterCondition>,
    /// String contains condition
    #[serde(default)]
    pub string_contains: Option<FilterCondition>,
    /// Starts with condition
    #[serde(default)]
    pub starts_with: Option<FilterCondition>,
}

/// Filter condition with key and value
#[derive(Debug, Clone, Deserialize)]
pub struct FilterCondition {
    pub key: String,
    pub value: serde_json::Value,
}

/// Retrieve response
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrieveResponse {
    /// Retrieved results
    pub retrieval_results: Vec<RetrievalResult>,
    /// Token for next page
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
}

/// Single retrieval result
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalResult {
    /// Content of the result
    pub content: RetrievalResultContent,
    /// Location information
    pub location: RetrievalResultLocation,
    /// Relevance score
    pub score: f32,
    /// Metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Content of a retrieval result
#[derive(Debug, Clone, Serialize)]
pub struct RetrievalResultContent {
    pub text: String,
}

/// Location of a retrieval result
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalResultLocation {
    #[serde(rename = "type")]
    pub location_type: String,
    /// S3 location (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub s3_location: Option<S3Location>,
    /// Custom location
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_location: Option<CustomLocation>,
}

/// S3 location
#[derive(Debug, Clone, Serialize)]
pub struct S3Location {
    pub uri: String,
}

/// Custom location for non-S3 sources
#[derive(Debug, Clone, Serialize)]
pub struct CustomLocation {
    pub uri: String,
}

/// Request to retrieve and generate
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrieveAndGenerateRequest {
    /// Input text/query
    pub input: RetrieveAndGenerateInput,
    /// Configuration
    pub retrieve_and_generate_configuration: RetrieveAndGenerateConfiguration,
    /// Session ID for conversation continuity
    #[serde(default)]
    pub session_id: Option<String>,
}

/// Input for retrieve and generate
#[derive(Debug, Clone, Deserialize)]
pub struct RetrieveAndGenerateInput {
    pub text: String,
}

/// Configuration for retrieve and generate
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrieveAndGenerateConfiguration {
    /// Type of RAG (KNOWLEDGE_BASE)
    #[serde(rename = "type")]
    pub config_type: String,
    /// Knowledge base configuration
    pub knowledge_base_configuration: KnowledgeBaseConfiguration,
}

/// Knowledge base configuration for RAG
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KnowledgeBaseConfiguration {
    /// Knowledge base ID
    pub knowledge_base_id: String,
    /// Model ARN (ignored, uses loaded model)
    #[serde(default)]
    pub model_arn: Option<String>,
    /// Retrieval configuration
    #[serde(default)]
    pub retrieval_configuration: Option<RetrievalConfiguration>,
    /// Generation configuration
    #[serde(default)]
    pub generation_configuration: Option<GenerationConfiguration>,
}

/// Generation configuration
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfiguration {
    /// Prompt template
    #[serde(default)]
    pub prompt_template: Option<PromptTemplate>,
    /// Inference configuration
    #[serde(default)]
    pub inference_config: Option<InferenceConfig>,
}

/// Prompt template
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptTemplate {
    /// Template text with $query$ and $search_results$ placeholders
    pub text_prompt_template: String,
}

/// Inference configuration
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceConfig {
    #[serde(default)]
    pub text_inference_config: Option<TextInferenceConfig>,
}

/// Text inference configuration
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextInferenceConfig {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

/// Retrieve and generate response
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrieveAndGenerateResponse {
    /// Generated output
    pub output: RetrieveAndGenerateOutput,
    /// Citations
    pub citations: Vec<Citation>,
    /// Session ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// Output from retrieve and generate
#[derive(Debug, Clone, Serialize)]
pub struct RetrieveAndGenerateOutput {
    pub text: String,
}

/// Citation in the response
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Citation {
    /// Text span in generated output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_response_part: Option<GeneratedResponsePart>,
    /// Retrieved references
    pub retrieved_references: Vec<RetrievedReference>,
}

/// Part of generated response being cited
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeneratedResponsePart {
    pub text_response_part: TextResponsePart,
}

/// Text part of response
#[derive(Debug, Clone, Serialize)]
pub struct TextResponsePart {
    pub text: String,
    pub span: Option<TextSpan>,
}

/// Span in text
#[derive(Debug, Clone, Serialize)]
pub struct TextSpan {
    pub start: usize,
    pub end: usize,
}

/// Retrieved reference
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievedReference {
    pub content: RetrievalResultContent,
    pub location: RetrievalResultLocation,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Request to ingest documents
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IngestRequest {
    /// Knowledge base ID
    pub knowledge_base_id: String,
    /// Documents to ingest
    pub documents: Vec<IngestDocument>,
}

/// Document to ingest
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IngestDocument {
    /// Document ID
    pub document_id: String,
    /// Content
    pub content: DocumentContent,
    /// Metadata
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

/// Document content
#[derive(Debug, Clone, Deserialize)]
pub struct DocumentContent {
    pub text: String,
}

/// Ingest response
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IngestResponse {
    /// Number of documents ingested
    pub documents_ingested: usize,
    /// Number of chunks created
    pub chunks_created: usize,
    /// Any failures
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub failures: Vec<IngestFailure>,
}

/// Ingest failure
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IngestFailure {
    pub document_id: String,
    pub error_message: String,
}

/// Request to list knowledge bases
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ListKnowledgeBasesRequest {
    #[serde(default)]
    pub max_results: Option<usize>,
    #[serde(default)]
    pub next_token: Option<String>,
}

/// Response listing knowledge bases
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ListKnowledgeBasesResponse {
    pub knowledge_base_summaries: Vec<KnowledgeBaseSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_token: Option<String>,
}

/// Knowledge base summary
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KnowledgeBaseSummary {
    pub knowledge_base_id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub status: String,
    pub updated_at: String,
}

/// Get knowledge base details response
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GetKnowledgeBaseResponse {
    pub knowledge_base: KnowledgeBaseDetail,
}

/// Knowledge base detail
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KnowledgeBaseDetail {
    pub knowledge_base_id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub status: String,
    pub storage_configuration: StorageConfigurationResponse,
    pub updated_at: String,
}

/// Storage configuration response
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageConfigurationResponse {
    #[serde(rename = "type")]
    pub storage_type: String,
    pub vector_dimension: usize,
}
