# pip install torch psutil transformers
import torch
import torch.nn.utils.prune as prune
import time
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def prune_model(model):
    # Prune 20% of weights in linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.2)
    return model

def quantize_model(model):
    # Dynamic quantization for better compatibility
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

def benchmark_inference(model, tokenizer, prompt="Hello world", num_runs=5):
    inputs = tokenizer(prompt, return_tensors="pt")
    start_time = time.time()
    memory_before = get_memory_usage()
    
    for _ in range(num_runs):
        with torch.no_grad():
            outputs = model(**inputs)
    
    end_time = time.time()
    memory_after = get_memory_usage()
    
    avg_time = (end_time - start_time) / num_runs
    memory_used = memory_after - memory_before
    return avg_time, memory_used

if __name__ == "__main__":
    # Load small pre-trained generative model
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    original_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Original model:")
    orig_time, orig_mem = benchmark_inference(original_model, tokenizer)
    print(f"  Inference time: {orig_time:.4f} s")
    print(f"  Memory usage: {orig_mem:.2f} MB")
    
    # Apply pruning to a copy
    pruned_model = AutoModelForCausalLM.from_pretrained(model_name)
    pruned_model = prune_model(pruned_model)
    print("\nAfter pruning:")
    prune_time, prune_mem = benchmark_inference(pruned_model, tokenizer)
    print(f"  Inference time: {prune_time:.4f} s")
    print(f"  Memory usage: {prune_mem:.2f} MB")
    
    # Apply quantization to original
    quantized_model = quantize_model(original_model)
    print("\nAfter quantization:")
    quant_time, quant_mem = benchmark_inference(quantized_model, tokenizer)
    print(f"  Inference time: {quant_time:.4f} s")
    print(f"  Memory usage: {quant_mem:.2f} MB")