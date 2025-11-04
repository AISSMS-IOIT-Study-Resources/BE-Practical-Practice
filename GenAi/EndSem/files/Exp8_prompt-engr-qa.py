# pip install transformers evaluate
from transformers import pipeline
from evaluate import load


def evaluate_prompt(prompt, reference):
    gen = pipeline('text-generation', model='distilgpt2', tokenizer='distilgpt2')
    out = gen(prompt, max_length=80, num_return_sequences=1)[0]['generated_text']

    # Basic ROUGE/ BLEU evaluation
    rouge = load('rouge')
    bleu = load('bleu')

    rouge_scores = rouge.compute(predictions=[out], references=[reference])
    # BLEU expects tokenized references
    bleu_scores = bleu.compute(predictions=[out], references=[reference])
    return {'generated': out, 'rouge': rouge_scores, 'bleu': bleu_scores}


# Example usage:
prompt = 'Summarize: The cat sat on the mat.'
reference = 'A cat was sitting on a mat.'
result = evaluate_prompt(prompt, reference)

print("Prompt:", prompt)
print("Reference:", reference)
print("\nGenerated Text:")
print(result['generated'])
print("\nROUGE Scores:")
for k, v in result['rouge'].items():
    print(f"  {k}: {v}")
print("\nBLEU Scores:")
for k, v in result['bleu'].items():
    print(f"  {k}: {v}")