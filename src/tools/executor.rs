//! Tool-call executor.
//!
//! Detects tool-call blocks in model output, executes them, and builds
//! the re-injection text that gets fed back into the conversation.

use super::filesystem::FilesystemTools;
use super::{ToolCall, ToolError, ToolRegistry};

const TOOL_CALL_OPEN: &str = "```tool_call";
const TOOL_CALL_CLOSE: &str = "```";

/// Maximum number of consecutive tool-call rounds before aborting.
const DEFAULT_MAX_ITERATIONS: usize = 16;

/// Configuration for the tool executor.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    pub max_iterations: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_MAX_ITERATIONS,
        }
    }
}

/// Executes tool calls detected in model output.
pub struct ToolExecutor {
    pub registry: ToolRegistry,
    pub fs_tools: Option<FilesystemTools>,
    pub config: ExecutorConfig,
}

impl ToolExecutor {
    pub fn new(
        registry: ToolRegistry,
        fs_tools: Option<FilesystemTools>,
        config: ExecutorConfig,
    ) -> Self {
        Self {
            registry,
            fs_tools,
            config,
        }
    }

    /// Try to extract a tool call from model output text.
    ///
    /// Looks for the pattern:
    /// ````text
    /// ```tool_call
    /// {"name": "tool_name", "arguments": {"param": "value"}}
    /// ```
    /// ````
    ///
    /// Returns `Some((text_before, tool_call, text_after))` if found, else `None`.
    pub fn parse_tool_call(text: &str) -> Option<(String, ToolCall, String)> {
        let open_idx = text.find(TOOL_CALL_OPEN)?;
        let json_start = open_idx + TOOL_CALL_OPEN.len();

        // Skip the optional newline after ```tool_call
        let remaining = &text[json_start..];
        let remaining = remaining.strip_prefix('\n').unwrap_or(remaining);
        let json_start_actual = text.len() - remaining.len();

        // Find the closing ```
        let close_idx = remaining.find(TOOL_CALL_CLOSE)?;
        let json_str = remaining[..close_idx].trim();

        let tool_call: ToolCall = serde_json::from_str(json_str).ok()?;

        let after_close = json_start_actual + close_idx + TOOL_CALL_CLOSE.len();
        let text_before = text[..open_idx].to_string();
        let text_after = if after_close < text.len() {
            text[after_close..].to_string()
        } else {
            String::new()
        };

        Some((text_before, tool_call, text_after))
    }

    /// Execute a single tool call and return the result as a formatted string
    /// suitable for re-injection into the conversation.
    pub fn execute_call(&self, call: &ToolCall) -> Result<String, ToolError> {
        // Check that the tool exists in the registry
        if self.registry.find(&call.name).is_none() {
            return Err(ToolError::UnknownTool(call.name.clone()));
        }

        // Dispatch to the appropriate handler
        let result = if let Some(ref fs) = self.fs_tools {
            match fs.execute(call) {
                Ok(r) => r,
                Err(e) => {
                    return Ok(format!(
                        "\n```tool_result\n{{\"tool\": \"{}\", \"success\": false, \"error\": \"{}\"}}\n```\n",
                        call.name,
                        e.to_string().replace('"', "\\\"")
                    ));
                }
            }
        } else {
            return Err(ToolError::UnknownTool(call.name.clone()));
        };

        let escaped_output = result.output.replace('"', "\\\"").replace('\n', "\\n");
        Ok(format!(
            "\n```tool_result\n{{\"tool\": \"{}\", \"success\": {}, \"output\": \"{}\"}}\n```\n",
            result.tool_name, result.success, escaped_output
        ))
    }

    /// Check if text contains a tool call.
    pub fn has_tool_call(text: &str) -> bool {
        text.contains(TOOL_CALL_OPEN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_call() {
        let text = "Let me read that file for you.\n\
                     ```tool_call\n\
                     {\"name\": \"read_file\", \"arguments\": {\"path\": \"/tmp/test.txt\"}}\n\
                     ```\n\
                     Now I'll process the result.";

        let (before, call, after) = ToolExecutor::parse_tool_call(text).unwrap();
        assert_eq!(before, "Let me read that file for you.\n");
        assert_eq!(call.name, "read_file");
        assert_eq!(
            call.arguments.get("path").unwrap().as_str().unwrap(),
            "/tmp/test.txt"
        );
        assert_eq!(after, "\nNow I'll process the result.");
    }

    #[test]
    fn test_no_tool_call() {
        let text = "Just a regular response with no tool calls.";
        assert!(ToolExecutor::parse_tool_call(text).is_none());
    }

    #[test]
    fn test_has_tool_call() {
        assert!(ToolExecutor::has_tool_call("text ```tool_call\n{}\n``` more"));
        assert!(!ToolExecutor::has_tool_call("just text"));
    }
}
