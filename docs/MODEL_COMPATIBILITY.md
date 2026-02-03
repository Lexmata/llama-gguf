# Model Compatibility

This document tracks GGUF models that have been tested with llama-rs.

## Supported Architectures

llama-rs currently supports models that use LLaMA-style tensor naming conventions:

| Architecture | Status | Notes |
|--------------|--------|-------|
| LLaMA | ✅ Supported | Original LLaMA, LLaMA 2, LLaMA 3 |
| Mistral | ✅ Verified | Mistral 7B and derivatives |
| Qwen/Qwen2 | ✅ Verified | Includes attention biases, uses NeoX-style RoPE |
| TinyLlama | ✅ Verified | LLaMA architecture with GQA |
| CodeLlama | ✅ Supported | Code-focused LLaMA variant |
| Yi | ✅ Supported | LLaMA-like architecture |
| DeepSeek | ✅ Verified | Uses linear RoPE scaling (scale_linear=4) |
| InternLM | 🔄 Untested | Uses LLaMA tensor names |
| Baichuan | 🔄 Untested | Uses LLaMA tensor names |
| Gemma2 | ❌ Not Supported | Different architecture with extra norm layers, logit softcapping |
| Phi-2 | ❌ Not Supported | Uses combined QKV tensors |
| GPT-NeoX | ❌ Not Supported | Uses combined QKV tensors |
| GPT-2 | ❌ Not Supported | Different architecture |
| BLOOM | ❌ Not Supported | Different architecture |
| Falcon | ❌ Not Supported | Different attention structure |
| MPT | ❌ Not Supported | Different architecture |

## Tested Models

### Successfully Tested Models

| Model | HuggingFace Repo | Size | Quantization | Load | Inference |
|-------|------------------|------|--------------|------|-----------|
| Qwen2.5-0.5B-Instruct | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` | 469 MB | Q4_K_M | ✅ | ✅ |
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct-GGUF` | 1.04 GB | Q4_K_M | ✅ | ✅ |
| TinyLlama-1.1B-Chat | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` | 638 MB | Q4_K_M | ✅ | ✅ |
| DeepSeek-Coder-1.3B | `TheBloke/deepseek-coder-1.3b-instruct-GGUF` | 833 MB | Q4_K_M | ✅ | ✅ |
| Mistral-7B-Instruct | `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` | 4.07 GB | Q4_K_M | ✅ | ✅ |

**Legend:**
- ✅ = Working correctly with correct output
- ⚠️ = Loads and runs but output quality varies (may be model-specific or quantization-dependent)
- ❌ = Not working

### Models That Fail to Load

| Model | HuggingFace Repo | Architecture | Error |
|-------|------------------|--------------|-------|
| Pythia-70M | `afrideva/pythia-70m-deduped-alpaca-cleaned-GGUF` | gptneox | Missing tensor: `blk.0.attn_q.weight` (uses combined QKV) |
| Phi-2 | `TheBloke/phi-2-GGUF` | phi2 | Missing tensor: `blk.0.attn_q.weight` (uses combined QKV) |
| Gemma-2-2B | `QuantFactory/gemma-2-2b-it-GGUF` | gemma2 | Shape mismatch (different architecture with extra norm layers) |

## Quantization Format Support

| Format | Status | Notes |
|--------|--------|-------|
| F32 | ✅ | Full precision |
| F16 | ✅ | Half precision |
| Q8_0 | ✅ | 8-bit quantization |
| Q8_1 | ✅ | 8-bit with improved accuracy |
| Q6_K | ✅ | 6-bit K-quant |
| Q5_0 | ✅ | 5-bit quantization |
| Q5_1 | ✅ | 5-bit with improved accuracy |
| Q5_K | ✅ | 5-bit K-quant |
| Q4_0 | ✅ | 4-bit quantization |
| Q4_1 | ✅ | 4-bit with improved accuracy |
| Q4_K | ✅ | 4-bit K-quant |
| Q3_K | ✅ | 3-bit K-quant |
| Q2_K | ✅ | 2-bit K-quant |

## Downloading Models

Use the built-in download command to fetch models from HuggingFace:

```bash
# List available files in a repository
llama-rs download Qwen/Qwen2.5-0.5B-Instruct-GGUF

# Download a specific file
llama-rs download Qwen/Qwen2.5-0.5B-Instruct-GGUF --file qwen2.5-0.5b-instruct-q4_k_m.gguf

# Search for models
llama-rs models search "llama gguf"

# List cached models
llama-rs models list
```

## Model Features

### Weight Tying

Models that share embedding weights with output projection (weight tying) are automatically detected. If `output.weight` tensor is missing, the token embedding weights are used for the output projection.

**Models using weight tying:**
- SmolLM series
- Some smaller models

### Attention Biases

Some model architectures include biases in attention projections:

**Models with attention biases:**
- Qwen/Qwen2 series (Q, K, V biases)

### Grouped Query Attention (GQA)

Models with fewer KV heads than query heads are supported:

| Model | Query Heads | KV Heads | GQA Ratio |
|-------|-------------|----------|-----------|
| TinyLlama-1.1B | 32 | 4 | 8:1 |
| Mistral-7B | 32 | 8 | 4:1 |
| Qwen2.5-0.5B | 14 | 2 | 7:1 |
| Qwen2.5-1.5B | 12 | 2 | 6:1 |
| DeepSeek-Coder-1.3B | 16 | 16 | 1:1 (MHA) |

## Known Issues

1. **GPT-NeoX/Phi Architecture**: Models using combined QKV projections (`attn_qkv.weight`) are not supported. This affects:
   - Pythia series
   - GPT-NeoX series
   - Phi-2
   - Some StableLM models

2. **Gemma2 Architecture**: Gemma2 models use additional features not yet implemented:
   - Extra normalization layers (`post_attention_norm`, `post_ffw_norm`)
   - Logit softcapping (attention and final logits)
   - Sliding window attention on alternating layers
   - Different head dimensions for K/V

3. **Instruction-Tuned Models**: Models like Mistral-7B-Instruct require proper prompt formatting (e.g., `[INST] ... [/INST]`) to produce high-quality output. Raw prompts like "1+1=" may not work well.

## Adding New Models

To test a new model:

1. Download using `llama-rs download <repo>`
2. Check model info with `llama-rs info <path>`
3. Verify architecture is LLaMA-compatible
4. Test inference with `llama-rs run <path> --prompt "test"`

Report compatibility results by opening an issue on GitHub.

## Verified Inference Results

### Qwen2.5-0.5B-Instruct (Q4_K_M)

```bash
$ llama-rs run qwen2.5-0.5b-instruct-q4_k_m.gguf -p "1+1=" -n 3 --temperature 0
1+1=2, 

$ llama-rs run qwen2.5-0.5b-instruct-q4_k_m.gguf -p "2+2=" -n 3 --temperature 0
2+2=4, 

$ llama-rs run qwen2.5-0.5b-instruct-q4_k_m.gguf -p "The capital of France is" -n 10 --temperature 0
The capital of France is____. A. Paris
```

### Qwen2.5-1.5B-Instruct (Q4_K_M)

```bash
$ llama-rs run qwen2.5-1.5b-instruct-q4_k_m.gguf -p "1+1=" -n 5 --temperature 0
1+1=2 is a basic principle

$ llama-rs run qwen2.5-1.5b-instruct-q4_k_m.gguf -p "The capital of Germany is" -n 10 --temperature 0
The capital of Germany is______. A. Berlin
```

Both Qwen models correctly predict arithmetic results and generate coherent text.

### TinyLlama-1.1B-Chat (Q4_K_M)

```bash
$ llama-rs run tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "1+1=" -n 8 --temperature 0
1+1=2.
The first number is the
```

### DeepSeek-Coder-1.3B (Q4_K_M)

```bash
$ llama-rs run deepseek-coder-1.3b-instruct.Q4_K_M.gguf -p "def fibonacci(n):" -n 30 --temperature 0
def fibonacci(n):
    if n <= 1 : 
   return (0, []) # base case for the recursion
```

### Mistral-7B-Instruct (Q4_K_M)

```bash
$ llama-rs run mistral-7b-instruct-v0.2.Q4_K_M.gguf -p "[INST] What is 1+1? [/INST]" -n 20 --temperature 0
[INST] What is 1+1? [/INST] The answer to the expression "1 + 1" is 2. In mathematics, this
```

Note: Mistral and other instruction-tuned models work best with proper prompt formatting.

## Test Environment

- **llama-rs version**: 0.1.0
- **Test date**: February 2026
- **Platform**: Linux x86_64
