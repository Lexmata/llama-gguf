#!/usr/bin/env python3
"""Check tokenizer mappings."""

from llama_cpp import Llama

def main():
    model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    
    llm = Llama(model_path=model_path, n_ctx=512, verbose=False)
    
    print("=== Token Mapping Check ===")
    print()
    
    # Check what token "2" maps to
    text_2 = "2"
    tokens_2 = llm.tokenize(text_2.encode(), add_bos=False)
    print(f"Text '{text_2}' -> Tokens: {tokens_2}")
    
    # Check token 17
    decoded_17 = llm.detokenize([17]).decode('utf-8', errors='replace')
    print(f"Token 17 decodes to: '{decoded_17}'")
    
    # Check the prompt tokenization
    prompt = "1+1="
    tokens = llm.tokenize(prompt.encode(), add_bos=False)
    print(f"\nPrompt '{prompt}' -> Tokens: {tokens}")
    
    for tok in tokens:
        decoded = llm.detokenize([tok]).decode('utf-8', errors='replace')
        print(f"  Token {tok} -> '{decoded}'")
    
    # Check a few other tokens
    print("\nOther token checks:")
    for i in [0, 1, 2, 10, 16, 17, 18, 28, 29, 30]:
        decoded = llm.detokenize([i]).decode('utf-8', errors='replace')
        print(f"  Token {i:3d} -> '{decoded}'")

if __name__ == "__main__":
    main()
