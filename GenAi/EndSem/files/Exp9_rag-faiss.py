# pip install faiss-cpu numpy transformers sentence-transformers
# Simple RAG-like example: build FAISS index over docs and use context + small LM to generate
# Requires: faiss, transformers

import faiss
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer


def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))
    return index

def retrieve_and_generate(query, doc_embeddings, query_embedding, texts, model_name='distilgpt2'):
    index = build_index(doc_embeddings)
    D, I = index.search(query_embedding.astype('float32').reshape(1, -1), 3)
    retrieved = [texts[i] for i in I[0]]
    context = '\n'.join(retrieved)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    gen = pipeline('text-generation', model=model_name, tokenizer=model_name)
    out = gen(prompt, max_length=50)[0]['generated_text']
    # Extract only the answer part after "Answer:"
    if "Answer:" in out:
        answer_part = out.split("Answer:", 1)[1].strip()
        # Take only the first line or sentence
        answer = answer_part.split('\n')[0].strip()
    else:
        answer = out
    return answer

# Note: you need an embedding model (e.g., sentence-transformers) to produce doc_embeddings and query_embedding.

if __name__ == "__main__":
    # Sample documents
    texts = [
        "The cat sat on the mat.",
        "Dogs are loyal animals.",
        "Machine learning is a subset of AI.",
        "Python is a programming language.",
        "FAISS is a library for similarity search."
    ]

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = model.encode(texts)

    # Query
    query = "What is machine learning?"
    query_embedding = model.encode([query])[0]

    # Generate answer
    answer = retrieve_and_generate(query, doc_embeddings, query_embedding, texts)
    print("Query:", query)
    print("Answer:", answer)