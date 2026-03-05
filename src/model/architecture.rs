//! Model architecture types

use serde::{Deserialize, Serialize};

/// Supported model architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Architecture {
    // LLaMA family
    Llama,
    Llama2,
    Llama3,

    // Mistral family
    Mistral,
    Mixtral,

    // Qwen family
    Qwen,
    Qwen2,
    Qwen2Moe,
    Qwen3,
    Qwen35,
    Qwen3Moe,
    Qwen3Next,

    // Phi family
    Phi,
    Phi2,
    Phi3,

    // Google
    Gemma,
    Gemma2,

    // Others
    Falcon,
    StarCoder,
    StarCoder2,
    MPT,
    GPTNeoX,
    GPT2,
    BLOOM,
    OPT,
    Baichuan,
    InternLM,
    InternLM2,
    Yi,
    DeepSeek,
    DeepSeekV2,
    CodeLlama,
    Orion,
    Olmo,
    Dbrx,
    Command,
    CommandR,
    Jamba,
    Mamba,

    /// Unknown architecture
    Unknown,
}

impl Architecture {
    /// Parse architecture from GGUF metadata string
    pub fn from_gguf_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "llama" => Self::Llama,
            "llama2" => Self::Llama2,
            "llama3" => Self::Llama3,
            "mistral" => Self::Mistral,
            "mixtral" => Self::Mixtral,
            "qwen" => Self::Qwen,
            "qwen2" => Self::Qwen2,
            "qwen2moe" => Self::Qwen2Moe,
            "qwen3" => Self::Qwen3,
            "qwen35" | "qwen3_5" | "qwen3.5" => Self::Qwen35,
            "qwen3moe" => Self::Qwen3Moe,
            "qwen3next" => Self::Qwen3Next,
            "phi" => Self::Phi,
            "phi2" => Self::Phi2,
            "phi3" => Self::Phi3,
            "gemma" => Self::Gemma,
            "gemma2" => Self::Gemma2,
            "falcon" => Self::Falcon,
            "starcoder" => Self::StarCoder,
            "starcoder2" => Self::StarCoder2,
            "mpt" => Self::MPT,
            "gptneox" | "gpt-neox" => Self::GPTNeoX,
            "gpt2" => Self::GPT2,
            "bloom" => Self::BLOOM,
            "opt" => Self::OPT,
            "baichuan" => Self::Baichuan,
            "internlm" => Self::InternLM,
            "internlm2" => Self::InternLM2,
            "yi" => Self::Yi,
            "deepseek" => Self::DeepSeek,
            "deepseek2" | "deepseekv2" => Self::DeepSeekV2,
            "codellama" | "code-llama" => Self::CodeLlama,
            "orion" => Self::Orion,
            "olmo" => Self::Olmo,
            "dbrx" => Self::Dbrx,
            "command" => Self::Command,
            "command-r" | "commandr" => Self::CommandR,
            "jamba" => Self::Jamba,
            "mamba" => Self::Mamba,
            _ => Self::Unknown,
        }
    }

    /// Get the architecture name as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Llama2 => "llama2",
            Self::Llama3 => "llama3",
            Self::Mistral => "mistral",
            Self::Mixtral => "mixtral",
            Self::Qwen => "qwen",
            Self::Qwen2 => "qwen2",
            Self::Qwen2Moe => "qwen2moe",
            Self::Qwen3 => "qwen3",
            Self::Qwen35 => "qwen35",
            Self::Qwen3Moe => "qwen3moe",
            Self::Qwen3Next => "qwen3next",
            Self::Phi => "phi",
            Self::Phi2 => "phi2",
            Self::Phi3 => "phi3",
            Self::Gemma => "gemma",
            Self::Gemma2 => "gemma2",
            Self::Falcon => "falcon",
            Self::StarCoder => "starcoder",
            Self::StarCoder2 => "starcoder2",
            Self::MPT => "mpt",
            Self::GPTNeoX => "gptneox",
            Self::GPT2 => "gpt2",
            Self::BLOOM => "bloom",
            Self::OPT => "opt",
            Self::Baichuan => "baichuan",
            Self::InternLM => "internlm",
            Self::InternLM2 => "internlm2",
            Self::Yi => "yi",
            Self::DeepSeek => "deepseek",
            Self::DeepSeekV2 => "deepseekv2",
            Self::CodeLlama => "codellama",
            Self::Orion => "orion",
            Self::Olmo => "olmo",
            Self::Dbrx => "dbrx",
            Self::Command => "command",
            Self::CommandR => "command-r",
            Self::Jamba => "jamba",
            Self::Mamba => "mamba",
            Self::Unknown => "unknown",
        }
    }

    /// Check if this architecture is LLaMA-like (uses same basic transformer structure)
    pub fn is_llama_like(&self) -> bool {
        matches!(
            self,
            Self::Llama
                | Self::Llama2
                | Self::Llama3
                | Self::Mistral
                | Self::CodeLlama
                | Self::Yi
                | Self::DeepSeek
                | Self::Orion
                | Self::Olmo
                | Self::Qwen
                | Self::Qwen2
                | Self::Qwen3
                | Self::Qwen35
                | Self::Qwen3Moe
                | Self::Qwen3Next
        )
    }

    /// Check if this architecture uses Mixture of Experts
    pub fn is_moe(&self) -> bool {
        matches!(
            self,
            Self::Mixtral
                | Self::Qwen2Moe
                | Self::Qwen3Moe
                | Self::Qwen3Next
                | Self::DeepSeekV2
                | Self::Dbrx
        )
    }

    /// Check if this architecture supports the standard LLaMA tensor naming scheme
    pub fn uses_llama_tensor_names(&self) -> bool {
        self.is_llama_like()
            || matches!(
                self,
                Self::Qwen
                    | Self::Qwen2
                    | Self::Qwen2Moe
                    | Self::Qwen3
                | Self::Qwen35
                    | Self::Qwen3Moe
                    | Self::Qwen3Next
                    | Self::InternLM
                    | Self::InternLM2
                    | Self::Baichuan
            )
    }

    /// Check if this architecture uses QK normalization before RoPE
    pub fn uses_qk_norm(&self) -> bool {
        matches!(self, Self::Qwen3 | Self::Qwen35 | Self::Qwen3Moe | Self::Qwen3Next)
    }
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_gguf_str() {
        assert_eq!(Architecture::from_gguf_str("llama"), Architecture::Llama);
        assert_eq!(Architecture::from_gguf_str("LLAMA"), Architecture::Llama);
        assert_eq!(
            Architecture::from_gguf_str("mistral"),
            Architecture::Mistral
        );
        assert_eq!(
            Architecture::from_gguf_str("unknown_arch"),
            Architecture::Unknown
        );
    }

    #[test]
    fn test_is_llama_like() {
        assert!(Architecture::Llama.is_llama_like());
        assert!(Architecture::Mistral.is_llama_like());
        assert!(!Architecture::GPT2.is_llama_like());
        assert!(!Architecture::BLOOM.is_llama_like());
    }
}
