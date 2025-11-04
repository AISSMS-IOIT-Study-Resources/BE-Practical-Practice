from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch, math

def evaluate_perplexity(model, tokenizer, text, device):
    """Evaluate model perplexity on given text"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = math.exp(loss.item())
    return loss.item(), perplexity

def generate_text(model, tokenizer, prompt, device, max_length=100):
    """Generate text using the model"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Small training corpus for fine-tuning
training_corpus = [
    "In a distant galaxy far away, space explorers discovered new worlds.",
    "The brave astronauts ventured into uncharted territories of the cosmos.",
    "Among the stars, ancient civilizations left mysterious artifacts.",
    "Interstellar travel opened new possibilities for human expansion.",
    "The spaceship navigated through asteroid fields with precision."
]

# LoRA Fine-tuning Implementation
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load base model
print("=== LOADING BASE MODEL ===")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)

# Evaluate base model
test_prompt = "In a distant galaxy,"
print(f"\n=== BASE MODEL EVALUATION ===")
base_text = generate_text(model, tokenizer, test_prompt, device)
print("Generated Text:", base_text)

# Calculate perplexity on training corpus
corpus_text = " ".join(training_corpus)
base_loss, base_perplexity = evaluate_perplexity(model, tokenizer, corpus_text, device)
print(f"Base Model - Loss: {base_loss:.4f}, Perplexity: {base_perplexity:.4f}")

# Setup LoRA configuration
print(f"\n=== APPLYING LoRA CONFIGURATION ===")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"]
)

# Apply LoRA
lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

# Simple fine-tuning on corpus (minimal training loop)
print(f"\n=== FINE-TUNING WITH LoRA ===")
lora_model.train()
optimizer = torch.optim.AdamW(lora_model.parameters(), lr=5e-5)

for epoch in range(3): 
    total_loss = 0
    for text in training_corpus:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        outputs = lora_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(training_corpus)
    print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

# Evaluate fine-tuned model
print(f"\n=== FINE-TUNED MODEL EVALUATION ===")
lora_model.eval()
finetuned_text = generate_text(lora_model, tokenizer, test_prompt, device)
print("Generated Text:", finetuned_text)

finetuned_loss, finetuned_perplexity = evaluate_perplexity(lora_model, tokenizer, corpus_text, device)
print(f"Fine-tuned Model - Loss: {finetuned_loss:.4f}, Perplexity: {finetuned_perplexity:.4f}")

print(f"\n=== COMPARISON SUMMARY ===")
print(f"Base Model      - Perplexity: {base_perplexity:.4f}")
print(f"Fine-tuned Model - Perplexity: {finetuned_perplexity:.4f}")
print(f"Improvement: {((base_perplexity - finetuned_perplexity) / base_perplexity * 100):.2f}%")