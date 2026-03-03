//! Filesystem tools with scoped access.
//!
//! All operations are restricted to a set of explicitly allowed directory trees.
//! Symlink traversal outside the allowed scope is rejected.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use super::{DeletePolicy, ToolCall, ToolDef, ToolError, ToolParam, ToolRegistry, ToolResult};

/// Filesystem scope restricting all operations to allowed directory trees.
#[derive(Debug, Clone)]
pub struct FilesystemScope {
    allowed_roots: Vec<PathBuf>,
}

impl FilesystemScope {
    pub fn new(allowed_roots: Vec<PathBuf>) -> Result<Self, ToolError> {
        let mut canonical = Vec::with_capacity(allowed_roots.len());
        for root in &allowed_roots {
            let c = root.canonicalize().map_err(|e| {
                ToolError::InvalidArguments(format!(
                    "Cannot resolve allowed path '{}': {}",
                    root.display(),
                    e
                ))
            })?;
            canonical.push(c);
        }
        Ok(Self {
            allowed_roots: canonical,
        })
    }

    /// Validate that a path falls within one of the allowed roots.
    /// Returns the canonicalized path on success.
    pub fn validate(&self, path: &Path) -> Result<PathBuf, ToolError> {
        // For paths that don't exist yet, canonicalize the parent and append the filename
        let canonical = if path.exists() {
            path.canonicalize().map_err(ToolError::Io)?
        } else {
            let parent = path
                .parent()
                .ok_or_else(|| ToolError::OutOfScope(path.display().to_string()))?;
            let parent_canon = parent.canonicalize().map_err(ToolError::Io)?;
            let filename = path
                .file_name()
                .ok_or_else(|| ToolError::InvalidArguments("No filename".into()))?;
            parent_canon.join(filename)
        };

        for root in &self.allowed_roots {
            if canonical.starts_with(root) {
                return Ok(canonical);
            }
        }

        Err(ToolError::OutOfScope(format!(
            "'{}' is outside allowed directories",
            path.display()
        )))
    }
}

/// Scoped filesystem handler that executes filesystem tool calls.
#[derive(Debug)]
pub struct FilesystemTools {
    scope: FilesystemScope,
    delete_policy: DeletePolicy,
}

impl FilesystemTools {
    pub fn new(scope: FilesystemScope, delete_policy: DeletePolicy) -> Self {
        Self {
            scope,
            delete_policy,
        }
    }

    /// Register all filesystem tools into the given registry.
    pub fn register_tools(registry: &mut ToolRegistry) {
        registry.register(ToolDef {
            name: "read_file".into(),
            description: "Read the contents of a file.".into(),
            parameters: vec![ToolParam {
                name: "path".into(),
                description: "Path to the file to read.".into(),
                param_type: "string".into(),
                required: true,
            }],
        });

        registry.register(ToolDef {
            name: "write_file".into(),
            description: "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.".into(),
            parameters: vec![
                ToolParam {
                    name: "path".into(),
                    description: "Path to the file to write.".into(),
                    param_type: "string".into(),
                    required: true,
                },
                ToolParam {
                    name: "content".into(),
                    description: "Content to write to the file.".into(),
                    param_type: "string".into(),
                    required: true,
                },
            ],
        });

        registry.register(ToolDef {
            name: "delete_file".into(),
            description: "Delete a file. May require confirmation depending on configuration.".into(),
            parameters: vec![ToolParam {
                name: "path".into(),
                description: "Path to the file to delete.".into(),
                param_type: "string".into(),
                required: true,
            }],
        });

        registry.register(ToolDef {
            name: "list_directory".into(),
            description: "List the contents of a directory.".into(),
            parameters: vec![ToolParam {
                name: "path".into(),
                description: "Path to the directory to list.".into(),
                param_type: "string".into(),
                required: true,
            }],
        });

        registry.register(ToolDef {
            name: "create_directory".into(),
            description: "Create a directory and any necessary parent directories.".into(),
            parameters: vec![ToolParam {
                name: "path".into(),
                description: "Path of the directory to create.".into(),
                param_type: "string".into(),
                required: true,
            }],
        });

        registry.register(ToolDef {
            name: "move_file".into(),
            description: "Move or rename a file or directory.".into(),
            parameters: vec![
                ToolParam {
                    name: "source".into(),
                    description: "Current path of the file or directory.".into(),
                    param_type: "string".into(),
                    required: true,
                },
                ToolParam {
                    name: "destination".into(),
                    description: "New path for the file or directory.".into(),
                    param_type: "string".into(),
                    required: true,
                },
            ],
        });

        registry.register(ToolDef {
            name: "file_info".into(),
            description: "Get information about a file (size, type, last modified time).".into(),
            parameters: vec![ToolParam {
                name: "path".into(),
                description: "Path to the file to inspect.".into(),
                param_type: "string".into(),
                required: true,
            }],
        });
    }

    /// Execute a filesystem tool call.
    pub fn execute(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        match call.name.as_str() {
            "read_file" => self.read_file(call),
            "write_file" => self.write_file(call),
            "delete_file" => self.delete_file(call),
            "list_directory" => self.list_directory(call),
            "create_directory" => self.create_directory(call),
            "move_file" => self.move_file(call),
            "file_info" => self.file_info(call),
            _ => Err(ToolError::UnknownTool(call.name.clone())),
        }
    }

    fn get_path_arg(&self, call: &ToolCall, name: &str) -> Result<PathBuf, ToolError> {
        let raw = call
            .arguments
            .get(name)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::InvalidArguments(format!("Missing required parameter: {}", name))
            })?;
        self.scope.validate(Path::new(raw))
    }

    fn get_string_arg(&self, call: &ToolCall, name: &str) -> Result<String, ToolError> {
        call.arguments
            .get(name)
            .and_then(|v| v.as_str())
            .map(String::from)
            .ok_or_else(|| {
                ToolError::InvalidArguments(format!("Missing required parameter: {}", name))
            })
    }

    fn read_file(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let path = self.get_path_arg(call, "path")?;
        let content = fs::read_to_string(&path)?;
        Ok(ToolResult {
            tool_name: "read_file".into(),
            success: true,
            output: content,
        })
    }

    fn write_file(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let path = self.get_path_arg(call, "path")?;
        let content = self.get_string_arg(call, "content")?;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, &content)?;
        Ok(ToolResult {
            tool_name: "write_file".into(),
            success: true,
            output: format!("Written {} bytes to {}", content.len(), path.display()),
        })
    }

    fn delete_file(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let path = self.get_path_arg(call, "path")?;

        match &self.delete_policy {
            DeletePolicy::AlwaysDeny => {
                return Err(ToolError::Denied(format!(
                    "Delete operations are not allowed. Path: {}",
                    path.display()
                )));
            }
            DeletePolicy::Confirm(callback) => {
                if !callback(&path) {
                    return Ok(ToolResult {
                        tool_name: "delete_file".into(),
                        success: false,
                        output: format!("Delete of '{}' was denied by user.", path.display()),
                    });
                }
            }
            DeletePolicy::AlwaysAllow => {}
        }

        if path.is_dir() {
            fs::remove_dir_all(&path)?;
        } else {
            fs::remove_file(&path)?;
        }
        Ok(ToolResult {
            tool_name: "delete_file".into(),
            success: true,
            output: format!("Deleted {}", path.display()),
        })
    }

    fn list_directory(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let path = self.get_path_arg(call, "path")?;
        let mut entries = Vec::new();

        for entry in fs::read_dir(&path)? {
            let entry = entry?;
            let ft = entry.file_type()?;
            let kind = if ft.is_dir() {
                "dir"
            } else if ft.is_symlink() {
                "symlink"
            } else {
                "file"
            };
            let name = entry.file_name().to_string_lossy().to_string();
            entries.push(format!("[{}] {}", kind, name));
        }

        entries.sort();
        Ok(ToolResult {
            tool_name: "list_directory".into(),
            success: true,
            output: if entries.is_empty() {
                "(empty directory)".into()
            } else {
                entries.join("\n")
            },
        })
    }

    fn create_directory(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let path = self.get_path_arg(call, "path")?;
        fs::create_dir_all(&path)?;
        Ok(ToolResult {
            tool_name: "create_directory".into(),
            success: true,
            output: format!("Created directory {}", path.display()),
        })
    }

    fn move_file(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let source = self.get_path_arg(call, "source")?;
        let destination = self.get_path_arg(call, "destination")?;

        // move_file is destructive for the source path
        match &self.delete_policy {
            DeletePolicy::AlwaysDeny => {
                return Err(ToolError::Denied(format!(
                    "Move operations are not allowed (destructive). Source: {}",
                    source.display()
                )));
            }
            DeletePolicy::Confirm(callback) => {
                if !callback(&source) {
                    return Ok(ToolResult {
                        tool_name: "move_file".into(),
                        success: false,
                        output: format!(
                            "Move of '{}' was denied by user.",
                            source.display()
                        ),
                    });
                }
            }
            DeletePolicy::AlwaysAllow => {}
        }

        fs::rename(&source, &destination)?;
        Ok(ToolResult {
            tool_name: "move_file".into(),
            success: true,
            output: format!(
                "Moved {} -> {}",
                source.display(),
                destination.display()
            ),
        })
    }

    fn file_info(&self, call: &ToolCall) -> Result<ToolResult, ToolError> {
        let path = self.get_path_arg(call, "path")?;
        let metadata = fs::metadata(&path)?;

        let kind = if metadata.is_dir() {
            "directory"
        } else if metadata.is_symlink() {
            "symlink"
        } else {
            "file"
        };

        let size = metadata.len();
        let modified = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
            .map(|d| chrono_lite(d.as_secs()))
            .unwrap_or_else(|| "unknown".into());

        let output = format!(
            "Type: {}\nSize: {} bytes\nLast modified: {}",
            kind, size, modified
        );

        Ok(ToolResult {
            tool_name: "file_info".into(),
            success: true,
            output,
        })
    }
}

/// Minimal unix-timestamp to human-readable date (avoids chrono dependency).
fn chrono_lite(epoch_secs: u64) -> String {
    let days = epoch_secs / 86400;
    let time_of_day = epoch_secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Compute year/month/day from days since epoch (1970-01-01)
    let mut y = 1970i64;
    let mut remaining = days as i64;

    loop {
        let days_in_year = if is_leap(y) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        y += 1;
    }

    let month_days: &[i64] = if is_leap(y) {
        &[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        &[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut m = 0;
    for &md in month_days {
        if remaining < md {
            break;
        }
        remaining -= md;
        m += 1;
    }

    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC",
        y,
        m + 1,
        remaining + 1,
        hours,
        minutes,
        seconds
    )
}

fn is_leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_scope_validation() {
        let dir = std::env::temp_dir().join("llama_fs_test_scope");
        fs::create_dir_all(&dir).unwrap();

        let scope = FilesystemScope::new(vec![dir.clone()]).unwrap();
        let valid = dir.join("test.txt");
        // Parent exists, file doesn't yet
        assert!(scope.validate(&valid).is_ok());

        let outside = Path::new("/etc/passwd");
        assert!(scope.validate(outside).is_err());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_read_write_roundtrip() {
        let dir = std::env::temp_dir().join("llama_fs_test_rw");
        fs::create_dir_all(&dir).unwrap();

        let scope = FilesystemScope::new(vec![dir.clone()]).unwrap();
        let tools = FilesystemTools::new(scope, DeletePolicy::AlwaysAllow);

        let file_path = dir.join("hello.txt");

        let write_call = ToolCall {
            name: "write_file".into(),
            arguments: HashMap::from([
                ("path".into(), serde_json::json!(file_path.to_str().unwrap())),
                ("content".into(), serde_json::json!("hello world")),
            ]),
        };
        let result = tools.execute(&write_call).unwrap();
        assert!(result.success);

        let read_call = ToolCall {
            name: "read_file".into(),
            arguments: HashMap::from([(
                "path".into(),
                serde_json::json!(file_path.to_str().unwrap()),
            )]),
        };
        let result = tools.execute(&read_call).unwrap();
        assert!(result.success);
        assert_eq!(result.output, "hello world");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_delete_denied() {
        let dir = std::env::temp_dir().join("llama_fs_test_deny");
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("doomed.txt");
        fs::write(&file, "data").unwrap();

        let scope = FilesystemScope::new(vec![dir.clone()]).unwrap();
        let tools = FilesystemTools::new(scope, DeletePolicy::AlwaysDeny);

        let call = ToolCall {
            name: "delete_file".into(),
            arguments: HashMap::from([(
                "path".into(),
                serde_json::json!(file.to_str().unwrap()),
            )]),
        };
        let result = tools.execute(&call);
        assert!(result.is_err());
        assert!(file.exists());

        fs::remove_dir_all(&dir).ok();
    }
}
