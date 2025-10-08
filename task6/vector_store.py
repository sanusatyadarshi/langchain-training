import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Initialize embeddings
print("=== Initializing Embeddings ===")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Test embedding
sample_text = "LangChain is a powerful framework for building AI applications"
sample_embedding = embeddings.embed_query(sample_text)
print(f"Sample text: {sample_text}")
print(f"Embedding dimensions: {len(sample_embedding)}")
print(f"First 5 dimensions: {sample_embedding[:5]}")

# Create sample documents
documents = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "intro", "category": "framework"}
    ),
    Document(
        page_content="LCEL (LangChain Expression Language) is a declarative way to compose chains.",
        metadata={"source": "lcel", "category": "syntax"}
    ),
    Document(
        page_content="Retrieval-Augmented Generation combines retrieval with text generation.",
        metadata={"source": "rag", "category": "technique"}
    ),
    Document(
        page_content="Vector stores enable semantic search by storing document embeddings.",
        metadata={"source": "vectors", "category": "storage"}
    ),
    Document(
        page_content="Prompt templates help create consistent and reusable prompts for LLMs.",
        metadata={"source": "prompts", "category": "templates"}
    ),
    Document(
        page_content="Memory systems allow chatbots to maintain conversation context.",
        metadata={"source": "memory", "category": "conversation"}
    ),
    Document(
        page_content="Agents can use tools to interact with external systems and APIs.",
        metadata={"source": "agents", "category": "automation"}
    ),
    Document(
        page_content="Document loaders help ingest text from various file formats and sources.",
        metadata={"source": "loaders", "category": "ingestion"}
    )
]

print(f"\n=== Creating Vector Store ===")
print(f"Processing {len(documents)} documents...")

# Create FAISS vector store
vector_store = FAISS.from_documents(documents, embeddings)
print("Vector store created successfully!")

# Save the vector store
vector_store.save_local("faiss_index")
print("Vector store saved to 'faiss_index' directory")

# Load vector store (demonstration)
loaded_vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
print("Vector store loaded successfully!")

# Similarity search demonstration
print("\n=== Similarity Search ===")

# Search for similar documents
query = "How do I create prompts for language models?"
results = vector_store.similarity_search(query, k=3)

print(f"Query: {query}")
print(f"Found {len(results)} similar documents:\n")

for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print()

# Similarity search with scores
print("=== Similarity Search with Scores ===")
results_with_scores = vector_store.similarity_search_with_score(query, k=3)

for i, (doc, score) in enumerate(results_with_scores):
    print(f"Result {i+1} (Score: {score:.4f}):")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print()

# Maximum Marginal Relevance (MMR) search
print("=== MMR Search (Diverse Results) ===")
mmr_results = vector_store.max_marginal_relevance_search(query, k=3)

print(f"MMR Query: {query}")
for i, doc in enumerate(mmr_results):
    print(f"MMR Result {i+1}: {doc.page_content}")

# Filter search by metadata
print("\n=== Filtered Search ===")
filter_results = vector_store.similarity_search(
    "framework",
    k=5,
    filter={"category": "framework"}
)

print("Results filtered by category='framework':")
for doc in filter_results:
    print(f"- {doc.page_content}")

# Add new documents to existing vector store
print("\n=== Adding New Documents ===")
new_docs = [
    Document(
        page_content="Text splitters break large documents into manageable chunks.",
        metadata={"source": "splitters", "category": "preprocessing"}
    ),
    Document(
        page_content="Output parsers structure the responses from language models.",
        metadata={"source": "parsers", "category": "postprocessing"}
    )
]

vector_store.add_documents(new_docs)
print(f"Added {len(new_docs)} new documents to vector store")

# Verify addition with a new search
verification_results = vector_store.similarity_search("text processing", k=2)
print("\nVerification search for 'text processing':")
for doc in verification_results:
    print(f"- {doc.page_content}")

with open('/root/vector-store.txt', 'w') as f:
    f.write("VECTOR_STORE_COMPLETE")