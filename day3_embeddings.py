from sentence_transformers import SentenceTransformer
import numpy as np

# Load a lightweight embedding model
# This runs locally — no API call, no cost, no tokens
model = SentenceTransformer('all-MiniLM-L6-v2')

# These are our sentences
sentences = [
    "How do I make an HTTP request in Python?",
    "What is a REST API?",
    "How do I use the requests library?",
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
]

# Convert sentences to vectors (embeddings)
embeddings = model.encode(sentences)

print(f"Each sentence becomes a vector of {len(embeddings[0])} numbers")
print(f"\nFirst embedding (first 5 numbers only):")
print(embeddings[0][:5])

# Now let's find which sentences are similar to a query
query = "How do I call an API in Python?"
query_embedding = model.encode(query)

# Calculate similarity between query and all sentences
# We use dot product — higher number = more similar
similarities = np.dot(embeddings, query_embedding)

print(f"\nQuery: '{query}'")
print("\nSimilarity scores:")

for sentence, score in zip(sentences, similarities):
    print(f"  {score:.3f} — {sentence}")

print("\nMost similar sentence:")
most_similar_idx = np.argmax(similarities)
print(f"  → {sentences[most_similar_idx]}")