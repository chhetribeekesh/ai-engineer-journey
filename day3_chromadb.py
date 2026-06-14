import chromadb
from sentence_transformers import SentenceTransformer

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB — this runs locally, no cloud, no cost
client = chromadb.Client()

# Create a collection — think of it like a table in a database
collection = client.create_collection(name="my_first_vectordb")

# Our documents — imagine these are chunks from a codebase or docs
documents = [
    "Python requests library is used to make HTTP calls to REST APIs",
    "ChromaDB is a vector database for storing and querying embeddings",
    "SQL injection happens when user input is directly concatenated into queries",
    "Docker containers package applications with their dependencies",
    "REST APIs use HTTP methods like GET POST PUT DELETE",
    "Git is a version control system for tracking code changes",
    "Vector embeddings convert text into numerical representations",
    "Authentication tokens should never be stored in plain text",
]

# Generate embeddings for all documents
embeddings = embedding_model.encode(documents).tolist()

# Store in ChromaDB
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"Stored {collection.count()} documents in ChromaDB")

# Now query it — just like a user asking a question
def search(query: str, n_results: int = 3):
    query_embedding = embedding_model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    print(f"\nQuery: '{query}'")
    print("Top results:")
    for i, doc in enumerate(results['documents'][0]):
        print(f"  {i+1}. {doc}")

# Test with different queries
search("how do I make API calls?")
search("what are security vulnerabilities I should avoid?")
search("how does version control work?")