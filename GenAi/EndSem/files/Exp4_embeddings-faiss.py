import numpy as np
import faiss
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

def load_corpus():
    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "I love natural language processing and machine learning",
        "Word embeddings help capture semantic relationships",
        "The fox is clever and quick",
        "Dogs are loyal and friendly animals",
        "Machine learning models improve with more data"
    ]
    tokenized = [simple_preprocess(doc) for doc in corpus]
    return tokenized

def train_word2vec(tokenized_corpus, vector_size=50, window=3, min_count=1):
    model = Word2Vec(
        sentences=tokenized_corpus, 
        vector_size=vector_size, 
        window=window, 
        min_count=min_count, 
        workers=1, 
        seed=42 
    )

    return model

def build_faiss_index(word_vectors):
    dim = word_vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(word_vectors)
    return index

def query_similar_words(model, index, word, top_k=5):
    if word not in model.wv:
        print(f"Word '{word}' not in vocabulary.")
        return
    
    query_vec = model.wv[word].reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query_vec, top_k + 1)

    print(f"\nTop {top_k} words similar to '{word}':")
    for dist, idx in zip(distances[0][1:], indices[0][1:]):
        similar_word = model.wv.index_to_key[idx]
        print(f" {similar_word} (distance: {dist:.4f})")

def main():
    tokenized_corpus = load_corpus()
    print("Tokenized corpus: ")
    for sent in tokenized_corpus:
        print(sent)
    
    model = train_word2vec(tokenized_corpus)
    word_vectors = model.wv.vectors.astype(np.float32)

    faiss_index = build_faiss_index(word_vectors)
    query_words = ['fox', 'machine', 'dog', 'language', 'quick']

    for word in query_words:
        query_similar_words(model, faiss_index, word)

main()