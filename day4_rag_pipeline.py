import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Same embedding model from Day 3 — must match for docs AND queries
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ChromaDB client — persistent, stores on disk this time (not memory)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="rag_pipeline_demo",
    metadata={"hnsw:space": "cosine"}
)
# Groq client for generation 
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

#A simple chunking function
def chunkText(text, chunk_size=300):
    """
    Simple chunking: splits on paragraph breaks (semantic-ish),
    falls back to fixed size if a paragraph is too long.
    """
    paragraphs = text.split("\n\n")
    chunks = []

    for lines in paragraphs:
        lines = lines.strip()
        if not lines:
            continue
        if len(lines) <= chunk_size:
            chunks.append(lines)
        else:
            # paragraph too long — fall back to fixed-size slicing
            for i in range(0,len(lines), chunk_size):
                chunks.append(lines[i:i+chunk_size])
    return chunks

# Indexing function (embed + store)
def index_documents(docs: list[dict]) -> None:
    """
    Indexing phase: chunk each doc, embed each chunk, store in ChromaDB.
    Runs once (or whenever docs change) — NOT on every query.
    """
    chunk_id = 0

    for doc in docs:
        chunks = chunkText(doc["text"])
        for chunk in chunks:
            embedding = embedder.encode(chunk).tolist()

            collection.add(
                ids=[f"chunk_{chunk_id}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": doc["source"]}]
            )
            chunk_id += 1
    print(f"Indexed {chunk_id} chunks from {len(docs)} documents.")

    #The query phase
def ask(question:str, threshold:float = 0.4, top_k: int = 3) -> str:
    """
    Query phase: embed question, retrieve top-k chunks,
    reject if below threshold, otherwise generate an answer.
    """

    querry_embedding= embedder.encode(question).tolist()

    results = collection.query(
        query_embeddings=[querry_embedding],
        n_results=top_k       
    )

    # ChromaDB returns distances — convert to similarity scores
    distances = results["distances"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    scored_chunks = []
    sources = []
    for doc, dist, meta in zip(documents, distances, metadatas):
        score = 1 - dist
        if score >= threshold:
            scored_chunks.append(doc)
            sources.append(meta["source"])
    
    if not scored_chunks:
        return "I don't know — I couldn't find relevant information in the indexed documents."

    context = "\n\n".join(scored_chunks)

    prompt = f"""Answer the question using ONLY the context below.
            If the answer isn't in the context, say "I don't know."

            Context:
            {context}

            Question: {question}

            Answer:"""
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    unique_sources = list(set(sources))
    answer = response.choices[0].message.content
    return f"{answer}\n\nSources: {', '.join(unique_sources)}"

if __name__ == "__main__":
    sample_docs = [
        {
            "source": "pricing.md",
            "text": "Stripe charges a 2.9% + 30 cents fee per transaction for standard accounts.\n\nEnterprise accounts get custom pricing negotiated with sales."
        },
        {
            "source": "refunds.md",
            "text": "Refunds can be issued within 90 days of the original charge.\n\nPartial refunds are supported and do not return the processing fee."
        }
    ]

    index_documents(sample_docs)
    
    while True:
        question = input("\nAsk a question (or 'quit'): ")
        if question.lower() == "quit":
            break
        answer = ask(question)
        print(f"\nAnswer: {answer}")
