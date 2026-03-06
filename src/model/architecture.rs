//! Model architecture types

use serde::{Deserialize, Serialize};

/// Supported model architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Architecture {
    // LLaMA family
    Llama,
    Llama2,
    Llama3,
    Llama4,

    // Mistral family
    Mistral,
    Mistral3,
    Mixtral,

    // Qwen family
    Qwen,
    Qwen2,
    Qwen2Moe,
    Qwen2VL,
    Qwen3,
    Qwen35,
    Qwen35Moe,
    Qwen3Moe,
    Qwen3Next,
    Qwen3VL,
    Qwen3VLMoe,

    // Phi family
    Phi,
    Phi2,
    Phi3,
    PhiMoe,

    // Google
    Gemma,
    Gemma2,
    Gemma3,
    Gemma3N,
    GemmaEmbedding,

    // Others
    Falcon,
    FalconH1,
    StarCoder,
    StarCoder2,
    MPT,
    GPTNeoX,
    GPT2,
    GPTJ,
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
    Olmo2,
    OlmoE,
    Dbrx,
    Command,
    CommandR,
    Jamba,
    Mamba,
    Mamba2,

    // Additional architectures from llama.cpp
    Deci,
    Grok,
    Refact,
    Bert,
    ModernBert,
    NomicBert,
    NomicBertMoe,
    NeoBert,
    JinaBertV2,
    JinaBertV3,
    EuroBert,
    StableLM,
    PLaMo,
    PLaMo2,
    PLaMo3,
    CodeShell,
    MiniCPM,
    MiniCPM3,
    XVerse,
    Cohere2,
    OpenELM,
    Arctic,
    ChatGLM,
    GLM4,
    GLM4Moe,
    GlmDsa,
    BitNet,
    T5,
    T5Encoder,
    JAIS,
    JAIS2,
    Nemotron,
    NemotronH,
    NemotronHMoe,
    ExaOne,
    ExaOne4,
    ExaOneMoe,
    RWKV6,
    RWKV6Qwen2,
    RWKV7,
    ARWKV7,
    Granite,
    GraniteMoe,
    GraniteHybrid,
    Chameleon,
    WavTokenizerDec,
    PLM,
    BailingMoe,
    BailingMoe2,
    Dots1,
    Arcee,
    AfMoe,
    Ernie45,
    Ernie45Moe,
    HunyuanMoe,
    HunyuanDense,
    SmolLM3,
    OpenAIMoe,
    LFM2,
    LFM2Moe,
    Dream,
    SmallThinker,
    LLaDA,
    LLaDAMoe,
    SeedOss,
    GroveMoe,
    Apertus,
    MinimaxM2,
    CogVLM,
    RND1,
    PanguEmbed,
    PaddleOCR,
    MIMO2,
    Step35,
    LlamaEmbed,
    MainCoder,
    KimiLinear,

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
            "llama4" => Self::Llama4,
            "mistral" => Self::Mistral,
            "mistral3" => Self::Mistral3,
            "mixtral" => Self::Mixtral,
            "qwen" => Self::Qwen,
            "qwen2" => Self::Qwen2,
            "qwen2moe" => Self::Qwen2Moe,
            "qwen2vl" => Self::Qwen2VL,
            "qwen3" => Self::Qwen3,
            "qwen35" => Self::Qwen35,
            "qwen35moe" => Self::Qwen35Moe,
            "qwen3moe" => Self::Qwen3Moe,
            "qwen3next" => Self::Qwen3Next,
            "qwen3vl" => Self::Qwen3VL,
            "qwen3vlmoe" => Self::Qwen3VLMoe,
            "phi" => Self::Phi,
            "phi2" => Self::Phi2,
            "phi3" => Self::Phi3,
            "phimoe" => Self::PhiMoe,
            "gemma" => Self::Gemma,
            "gemma2" => Self::Gemma2,
            "gemma3" => Self::Gemma3,
            "gemma3n" => Self::Gemma3N,
            "gemma-embedding" => Self::GemmaEmbedding,
            "falcon" => Self::Falcon,
            "falcon-h1" => Self::FalconH1,
            "starcoder" => Self::StarCoder,
            "starcoder2" => Self::StarCoder2,
            "mpt" => Self::MPT,
            "gptneox" | "gpt-neox" => Self::GPTNeoX,
            "gpt2" => Self::GPT2,
            "gptj" | "gpt-j" => Self::GPTJ,
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
            "olmo2" => Self::Olmo2,
            "olmoe" => Self::OlmoE,
            "dbrx" => Self::Dbrx,
            "command" => Self::Command,
            "command-r" | "commandr" => Self::CommandR,
            "jamba" => Self::Jamba,
            "mamba" => Self::Mamba,
            "mamba2" => Self::Mamba2,
            "deci" => Self::Deci,
            "grok" => Self::Grok,
            "refact" => Self::Refact,
            "bert" => Self::Bert,
            "modern-bert" => Self::ModernBert,
            "nomic-bert" => Self::NomicBert,
            "nomic-bert-moe" => Self::NomicBertMoe,
            "neo-bert" => Self::NeoBert,
            "jina-bert-v2" => Self::JinaBertV2,
            "jina-bert-v3" => Self::JinaBertV3,
            "eurobert" => Self::EuroBert,
            "stablelm" => Self::StableLM,
            "plamo" => Self::PLaMo,
            "plamo2" => Self::PLaMo2,
            "plamo3" => Self::PLaMo3,
            "codeshell" => Self::CodeShell,
            "minicpm" => Self::MiniCPM,
            "minicpm3" => Self::MiniCPM3,
            "xverse" => Self::XVerse,
            "cohere2" => Self::Cohere2,
            "openelm" => Self::OpenELM,
            "arctic" => Self::Arctic,
            "chatglm" => Self::ChatGLM,
            "glm4" => Self::GLM4,
            "glm4moe" => Self::GLM4Moe,
            "glm-dsa" => Self::GlmDsa,
            "bitnet" => Self::BitNet,
            "t5" => Self::T5,
            "t5encoder" => Self::T5Encoder,
            "jais" => Self::JAIS,
            "jais2" => Self::JAIS2,
            "nemotron" => Self::Nemotron,
            "nemotron_h" => Self::NemotronH,
            "nemotron_h_moe" => Self::NemotronHMoe,
            "exaone" => Self::ExaOne,
            "exaone4" => Self::ExaOne4,
            "exaone-moe" => Self::ExaOneMoe,
            "rwkv6" => Self::RWKV6,
            "rwkv6qwen2" => Self::RWKV6Qwen2,
            "rwkv7" => Self::RWKV7,
            "arwkv7" => Self::ARWKV7,
            "granite" => Self::Granite,
            "granitemoe" => Self::GraniteMoe,
            "granitehybrid" => Self::GraniteHybrid,
            "chameleon" => Self::Chameleon,
            "wavtokenizer-dec" => Self::WavTokenizerDec,
            "plm" => Self::PLM,
            "bailingmoe" => Self::BailingMoe,
            "bailingmoe2" => Self::BailingMoe2,
            "dots1" => Self::Dots1,
            "arcee" => Self::Arcee,
            "afmoe" => Self::AfMoe,
            "ernie4_5" => Self::Ernie45,
            "ernie4_5-moe" => Self::Ernie45Moe,
            "hunyuan-moe" => Self::HunyuanMoe,
            "hunyuan-dense" => Self::HunyuanDense,
            "smollm3" => Self::SmolLM3,
            "gpt-oss" => Self::OpenAIMoe,
            "lfm2" => Self::LFM2,
            "lfm2moe" => Self::LFM2Moe,
            "dream" => Self::Dream,
            "smallthinker" => Self::SmallThinker,
            "llada" => Self::LLaDA,
            "llada-moe" => Self::LLaDAMoe,
            "seed_oss" => Self::SeedOss,
            "grovemoe" => Self::GroveMoe,
            "apertus" => Self::Apertus,
            "minimax-m2" => Self::MinimaxM2,
            "cogvlm" => Self::CogVLM,
            "rnd1" => Self::RND1,
            "pangu-embedded" => Self::PanguEmbed,
            "paddleocr" => Self::PaddleOCR,
            "mimo2" => Self::MIMO2,
            "step35" => Self::Step35,
            "llama-embed" => Self::LlamaEmbed,
            "maincoder" => Self::MainCoder,
            "kimi-linear" => Self::KimiLinear,
            _ => Self::Unknown,
        }
    }

    /// Get the architecture name as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Llama2 => "llama2",
            Self::Llama3 => "llama3",
            Self::Llama4 => "llama4",
            Self::Mistral => "mistral",
            Self::Mistral3 => "mistral3",
            Self::Mixtral => "mixtral",
            Self::Qwen => "qwen",
            Self::Qwen2 => "qwen2",
            Self::Qwen2Moe => "qwen2moe",
            Self::Qwen2VL => "qwen2vl",
            Self::Qwen3 => "qwen3",
            Self::Qwen35 => "qwen35",
            Self::Qwen35Moe => "qwen35moe",
            Self::Qwen3Moe => "qwen3moe",
            Self::Qwen3Next => "qwen3next",
            Self::Qwen3VL => "qwen3vl",
            Self::Qwen3VLMoe => "qwen3vlmoe",
            Self::Phi => "phi",
            Self::Phi2 => "phi2",
            Self::Phi3 => "phi3",
            Self::PhiMoe => "phimoe",
            Self::Gemma => "gemma",
            Self::Gemma2 => "gemma2",
            Self::Gemma3 => "gemma3",
            Self::Gemma3N => "gemma3n",
            Self::GemmaEmbedding => "gemma-embedding",
            Self::Falcon => "falcon",
            Self::FalconH1 => "falcon-h1",
            Self::StarCoder => "starcoder",
            Self::StarCoder2 => "starcoder2",
            Self::MPT => "mpt",
            Self::GPTNeoX => "gptneox",
            Self::GPT2 => "gpt2",
            Self::GPTJ => "gptj",
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
            Self::Olmo2 => "olmo2",
            Self::OlmoE => "olmoe",
            Self::Dbrx => "dbrx",
            Self::Command => "command",
            Self::CommandR => "command-r",
            Self::Jamba => "jamba",
            Self::Mamba => "mamba",
            Self::Mamba2 => "mamba2",
            Self::Deci => "deci",
            Self::Grok => "grok",
            Self::Refact => "refact",
            Self::Bert => "bert",
            Self::ModernBert => "modern-bert",
            Self::NomicBert => "nomic-bert",
            Self::NomicBertMoe => "nomic-bert-moe",
            Self::NeoBert => "neo-bert",
            Self::JinaBertV2 => "jina-bert-v2",
            Self::JinaBertV3 => "jina-bert-v3",
            Self::EuroBert => "eurobert",
            Self::StableLM => "stablelm",
            Self::PLaMo => "plamo",
            Self::PLaMo2 => "plamo2",
            Self::PLaMo3 => "plamo3",
            Self::CodeShell => "codeshell",
            Self::MiniCPM => "minicpm",
            Self::MiniCPM3 => "minicpm3",
            Self::XVerse => "xverse",
            Self::Cohere2 => "cohere2",
            Self::OpenELM => "openelm",
            Self::Arctic => "arctic",
            Self::ChatGLM => "chatglm",
            Self::GLM4 => "glm4",
            Self::GLM4Moe => "glm4moe",
            Self::GlmDsa => "glm-dsa",
            Self::BitNet => "bitnet",
            Self::T5 => "t5",
            Self::T5Encoder => "t5encoder",
            Self::JAIS => "jais",
            Self::JAIS2 => "jais2",
            Self::Nemotron => "nemotron",
            Self::NemotronH => "nemotron_h",
            Self::NemotronHMoe => "nemotron_h_moe",
            Self::ExaOne => "exaone",
            Self::ExaOne4 => "exaone4",
            Self::ExaOneMoe => "exaone-moe",
            Self::RWKV6 => "rwkv6",
            Self::RWKV6Qwen2 => "rwkv6qwen2",
            Self::RWKV7 => "rwkv7",
            Self::ARWKV7 => "arwkv7",
            Self::Granite => "granite",
            Self::GraniteMoe => "granitemoe",
            Self::GraniteHybrid => "granitehybrid",
            Self::Chameleon => "chameleon",
            Self::WavTokenizerDec => "wavtokenizer-dec",
            Self::PLM => "plm",
            Self::BailingMoe => "bailingmoe",
            Self::BailingMoe2 => "bailingmoe2",
            Self::Dots1 => "dots1",
            Self::Arcee => "arcee",
            Self::AfMoe => "afmoe",
            Self::Ernie45 => "ernie4_5",
            Self::Ernie45Moe => "ernie4_5-moe",
            Self::HunyuanMoe => "hunyuan-moe",
            Self::HunyuanDense => "hunyuan-dense",
            Self::SmolLM3 => "smollm3",
            Self::OpenAIMoe => "gpt-oss",
            Self::LFM2 => "lfm2",
            Self::LFM2Moe => "lfm2moe",
            Self::Dream => "dream",
            Self::SmallThinker => "smallthinker",
            Self::LLaDA => "llada",
            Self::LLaDAMoe => "llada-moe",
            Self::SeedOss => "seed_oss",
            Self::GroveMoe => "grovemoe",
            Self::Apertus => "apertus",
            Self::MinimaxM2 => "minimax-m2",
            Self::CogVLM => "cogvlm",
            Self::RND1 => "rnd1",
            Self::PanguEmbed => "pangu-embedded",
            Self::PaddleOCR => "paddleocr",
            Self::MIMO2 => "mimo2",
            Self::Step35 => "step35",
            Self::LlamaEmbed => "llama-embed",
            Self::MainCoder => "maincoder",
            Self::KimiLinear => "kimi-linear",
            Self::Unknown => "unknown",
        }
    }

    /// Check if this architecture is LLaMA-like (uses same basic transformer structure)
    pub fn is_llama_like(&self) -> bool {
        !self.is_encoder_only()
            && !self.is_recurrent()
            && !matches!(self, Self::Unknown | Self::T5 | Self::T5Encoder)
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
                | Self::PhiMoe
                | Self::Qwen35Moe
                | Self::Qwen3VLMoe
                | Self::OlmoE
                | Self::Arctic
                | Self::GLM4Moe
                | Self::NemotronHMoe
                | Self::ExaOneMoe
                | Self::GraniteMoe
                | Self::NomicBertMoe
                | Self::BailingMoe
                | Self::BailingMoe2
                | Self::AfMoe
                | Self::Ernie45Moe
                | Self::HunyuanMoe
                | Self::OpenAIMoe
                | Self::LFM2Moe
                | Self::LLaDAMoe
                | Self::GroveMoe
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
                    | Self::Qwen3Moe
                    | Self::Qwen3Next
                    | Self::InternLM
                    | Self::InternLM2
                    | Self::Baichuan
                    | Self::Mixtral
                    | Self::Deci
                    | Self::Llama4
                    | Self::Cohere2
                    | Self::CommandR
                    | Self::Command
                    | Self::Nemotron
                    | Self::ExaOne
                    | Self::ExaOne4
                    | Self::Granite
                    | Self::GraniteMoe
                    | Self::JAIS
                    | Self::JAIS2
                    | Self::Arcee
                    | Self::SmolLM3
                    | Self::Mistral3
                    | Self::XVerse
                    | Self::MiniCPM
                    | Self::MiniCPM3
                    | Self::StableLM
            )
    }

    /// Check if this architecture uses QK normalization before RoPE
    pub fn uses_qk_norm(&self) -> bool {
        matches!(
            self,
            Self::Qwen3 | Self::Qwen3Moe | Self::Qwen3Next | Self::Gemma2 | Self::Gemma3
                | Self::Gemma3N
                | Self::Cohere2
                | Self::Olmo2
        )
    }

    /// Check if this architecture uses a fused QKV tensor
    pub fn has_combined_qkv(&self) -> bool {
        matches!(
            self,
            Self::Phi2
                | Self::Phi3
                | Self::PhiMoe
                | Self::GPTNeoX
                | Self::GPTJ
                | Self::Falcon
                | Self::FalconH1
                | Self::StableLM
                | Self::Refact
                | Self::CodeShell
                | Self::BLOOM
                | Self::MPT
        )
    }

    /// Check if this architecture has post-FFN normalization
    pub fn has_post_ffn_norm(&self) -> bool {
        matches!(
            self,
            Self::Gemma2 | Self::Gemma3 | Self::Gemma3N | Self::Cohere2 | Self::Olmo2
        )
    }

    /// Check if this architecture uses logit softcapping
    pub fn has_logit_softcap(&self) -> bool {
        matches!(self, Self::Gemma2)
    }

    /// Check if this architecture is in the Gemma family (uses weight+1 in RMS norm)
    pub fn is_gemma(&self) -> bool {
        matches!(
            self,
            Self::Gemma
                | Self::Gemma2
                | Self::Gemma3
                | Self::Gemma3N
                | Self::GemmaEmbedding
        )
    }

    /// Check if this architecture uses LayerNorm (not RMSNorm)
    pub fn uses_layer_norm(&self) -> bool {
        matches!(
            self,
            Self::GPT2
                | Self::BLOOM
                | Self::GPTJ
                | Self::GPTNeoX
                | Self::Falcon
                | Self::MPT
                | Self::OPT
                | Self::Phi2
                | Self::Bert
                | Self::ModernBert
                | Self::NomicBert
                | Self::JinaBertV2
                | Self::JinaBertV3
                | Self::EuroBert
                | Self::T5
                | Self::T5Encoder
                | Self::StableLM
        )
    }

    /// Check if this architecture uses GELU activation
    pub fn uses_gelu(&self) -> bool {
        matches!(
            self,
            Self::Phi2
                | Self::Phi3
                | Self::PhiMoe
                | Self::GPT2
                | Self::GPTJ
                | Self::GPTNeoX
                | Self::BLOOM
                | Self::OPT
                | Self::Falcon
                | Self::StableLM
                | Self::Bert
                | Self::ModernBert
                | Self::StarCoder
                | Self::StarCoder2
                | Self::CodeShell
                | Self::Refact
        )
    }

    /// Check if this architecture is encoder-only (no causal LM head)
    pub fn is_encoder_only(&self) -> bool {
        matches!(
            self,
            Self::Bert
                | Self::ModernBert
                | Self::NomicBert
                | Self::NomicBertMoe
                | Self::NeoBert
                | Self::JinaBertV2
                | Self::JinaBertV3
                | Self::EuroBert
                | Self::GemmaEmbedding
                | Self::LlamaEmbed
                | Self::PanguEmbed
        )
    }

    /// Check if this architecture is recurrent (SSM/RWKV style)
    pub fn is_recurrent(&self) -> bool {
        matches!(
            self,
            Self::Mamba
                | Self::Mamba2
                | Self::RWKV6
                | Self::RWKV7
                | Self::ARWKV7
                | Self::RWKV6Qwen2
        )
    }

    /// Check if this architecture has 2-projection FFN (no gate)
    pub fn has_no_gate_ffn(&self) -> bool {
        matches!(
            self,
            Self::GPT2
                | Self::GPTJ
                | Self::GPTNeoX
                | Self::BLOOM
                | Self::OPT
                | Self::Falcon
                | Self::Phi
                | Self::Phi2
                | Self::Phi3
                | Self::PhiMoe
                | Self::StableLM
                | Self::CodeShell
                | Self::Bert
                | Self::ModernBert
                | Self::NomicBert
                | Self::T5
                | Self::T5Encoder
        )
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
            Architecture::from_gguf_str("llama4"),
            Architecture::Llama4
        );
        assert_eq!(
            Architecture::from_gguf_str("gptj"),
            Architecture::GPTJ
        );
        assert_eq!(
            Architecture::from_gguf_str("gpt-j"),
            Architecture::GPTJ
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
        assert!(Architecture::Llama4.is_llama_like());
        assert!(Architecture::MiniCPM3.is_llama_like());
        assert!(Architecture::GPT2.is_llama_like());
        assert!(Architecture::Falcon.is_llama_like());
        assert!(Architecture::Phi2.is_llama_like());
        assert!(Architecture::Gemma2.is_llama_like());
        assert!(!Architecture::Bert.is_llama_like());
        assert!(!Architecture::Mamba.is_llama_like());
    }
}
