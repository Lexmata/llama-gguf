#!/usr/bin/env python3
"""Use llama-cpp-python generation to see what it predicts for various inputs."""

from llama_cpp import Llama

def main():
    model_path = "/home/joseph/Models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    
    print("Loading model...")
    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        verbose=False,
    )
    
    test_prompts = [
        "=",
        "1=",
        "+=",
        "a=",
        " =",
        "==",
        "===",
        "1+1=",
        "2+2=",
        "11",
        "12",
    ]
    
    print()
    print("Prompt      | Generated | Expected")
    print("------------+-----------+----------")
    
    for prompt in test_prompts:
        output = llm(prompt, max_tokens=1, temperature=0.0, top_k=1)
        generated = output['choices'][0]['text'].strip()
        
        # What we might expect
        expected = "?"
        if prompt == "1+1=":
            expected = "2"
        elif prompt == "2+2=":
            expected = "4"
        elif prompt.endswith("="):
            expected = "varies"
        
        print(f"{prompt:11} | {generated:9} | {expected}")

if __name__ == "__main__":
    main()
