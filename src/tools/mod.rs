//! LLM tool-calling framework.
//!
//! Provides a registry of tools that models can invoke during generation.
//! Tools are described in JSON schema format and injected into the system prompt.
//! The model emits tool calls as structured blocks which the executor parses,
//! validates, runs, and feeds back into the conversation.

pub mod executor;
pub mod filesystem;

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

/// Description of a single tool parameter.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolParam {
    pub name: String,
    pub description: String,
    pub param_type: String,
    pub required: bool,
}

/// Definition of a tool the model can call.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParam>,
}

/// A parsed tool call from model output.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: HashMap<String, serde_json::Value>,
}

/// Result of executing a tool call.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_name: String,
    pub success: bool,
    pub output: String,
}

impl fmt::Display for ToolResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.success {
            write!(f, "{}", self.output)
        } else {
            write!(f, "Error: {}", self.output)
        }
    }
}

/// Policy for handling destructive operations (delete, move).
#[derive(Clone)]
pub enum DeletePolicy {
    /// Always ask for confirmation via the provided callback.
    /// The callback receives the path and returns `true` to proceed.
    Confirm(std::sync::Arc<dyn Fn(&Path) -> bool + Send + Sync>),
    /// Allow all destructive operations without confirmation.
    AlwaysAllow,
    /// Deny all destructive operations.
    AlwaysDeny,
}

impl fmt::Debug for DeletePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeletePolicy::Confirm(_) => write!(f, "Confirm(callback)"),
            DeletePolicy::AlwaysAllow => write!(f, "AlwaysAllow"),
            DeletePolicy::AlwaysDeny => write!(f, "AlwaysDeny"),
        }
    }
}

// DeletePolicy::AlwaysDeny is the safest default — no file deletions without explicit opt-in.
#[allow(clippy::derivable_impls)]
impl Default for DeletePolicy {
    fn default() -> Self {
        DeletePolicy::AlwaysDeny
    }
}

/// Error type for tool operations.
#[derive(thiserror::Error, Debug)]
pub enum ToolError {
    #[error("Path outside allowed scope: {0}")]
    OutOfScope(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Operation denied: {0}")]
    Denied(String),

    #[error("Invalid arguments: {0}")]
    InvalidArguments(String),

    #[error("Unknown tool: {0}")]
    UnknownTool(String),

    #[error("Tool call parse error: {0}")]
    ParseError(String),

    #[error("Max tool iterations exceeded")]
    MaxIterationsExceeded,
}

pub type ToolResult2 = std::result::Result<ToolResult, ToolError>;

/// Registry of available tools.
#[derive(Debug)]
pub struct ToolRegistry {
    tools: Vec<ToolDef>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register(&mut self, tool: ToolDef) {
        self.tools.push(tool);
    }

    pub fn tools(&self) -> &[ToolDef] {
        &self.tools
    }

    pub fn find(&self, name: &str) -> Option<&ToolDef> {
        self.tools.iter().find(|t| t.name == name)
    }

    /// Generate a system prompt section describing all available tools.
    pub fn system_prompt_section(&self) -> String {
        if self.tools.is_empty() {
            return String::new();
        }

        let mut s = String::from(
            "\n\nYou have access to the following tools. To use a tool, output a tool call block:\n\n\
             ```tool_call\n\
             {\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n\
             ```\n\n\
             Wait for the tool result before continuing. Available tools:\n\n",
        );

        for tool in &self.tools {
            s.push_str(&format!("### {}\n{}\n", tool.name, tool.description));
            if !tool.parameters.is_empty() {
                s.push_str("Parameters:\n");
                for p in &tool.parameters {
                    let req = if p.required { " (required)" } else { "" };
                    s.push_str(&format!(
                        "- `{}` ({}{}): {}\n",
                        p.name, p.param_type, req, p.description
                    ));
                }
            }
            s.push('\n');
        }

        s
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
