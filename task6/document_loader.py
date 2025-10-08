from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Sample documents for demonstration
sample_docs = [
    "LangChain is a framework for developing applications powered by language models. It enables applications that are data-aware and agentic, allowing language models to connect with other sources of data and interact with their environment.",

    "The LangChain Expression Language (LCEL) is a declarative way to compose chains. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest 'prompt + LLM' chain to the most complex chains.",

    "Retrieval-Augmented Generation (RAG) is a technique that combines retrieval of relevant documents with generation of responses. This allows language models to access external knowledge beyond their training data.",

    "Vector stores are databases optimized for storing and searching high-dimensional vectors. In the context of RAG, they store embeddings of documents that can be efficiently searched for semantic similarity.",

    "Prompt templates are a powerful way to create reusable prompts for language models. They allow you to parameterize prompts and create structured inputs for consistent model behavior."
]

print("=== Document Loading and Chunking ===")

# Convert strings to Document objects
documents = [
    Document(
        page_content=content,
        metadata={"source": f"doc_{i}", "type": "educational"}
    )
    for i, content in enumerate(sample_docs)
]

print(f"Created {len(documents)} documents")
for i, doc in enumerate(documents):
    print(f"Doc {i}: {doc.page_content[:100]}...")

# Text splitting demonstration
print("\n=== Text Splitting ===")

# Large document for splitting
large_doc = """
LangChain is a framework for developing applications powered by language models. It enables applications that are data-aware and agentic, allowing language models to connect with other sources of data and interact with their environment.

The main value props of LangChain are:
1. Components: abstractions for working with language models
2. Off-the-shelf chains: a structured assembly of components for accomplishing specific higher-level tasks

Off-the-shelf chains make it easy to get started. For more complex applications and nuanced use-cases, components make it easy to customize existing chains or build new ones.

The LangChain Expression Language (LCEL) is a declarative way to compose chains. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest "prompt + LLM" chain to the most complex chains.

LCEL has a number of benefits:
- Streaming support
- Async support
- Optimized parallel execution
- Retries and fallbacks
- Access to intermediate results
- Input and output schemas
"""

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

# Split the large document
chunks = text_splitter.split_text(large_doc)
print(f"Split into {len(chunks)} chunks:")

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(chunk)

# Create Document objects from chunks
chunk_documents = [
    Document(
        page_content=chunk,
        metadata={
            "source": "langchain_overview",
            "chunk": i,
            "total_chunks": len(chunks)
        }
    )
    for i, chunk in enumerate(chunks)
]

print(f"\n=== Document Objects Created ===")
print(f"Total chunk documents: {len(chunk_documents)}")

# Best practices demonstration
print("\n=== Best Practices ===")

# Different splitter configurations
print("1. Code-specific splitter:")
code_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", ""]
)

sample_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def main():
    result = calculate_fibonacci(10)
    print(f"Fibonacci(10) = {result}")

if __name__ == "__main__":
    main()
"""

code_chunks = code_splitter.split_text(sample_code)
print(f"Code split into {len(code_chunks)} chunks")

print("\n2. Markdown-aware splitter:")
markdown_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    separators=["## ", "# ", "\n\n", "\n", " ", ""]
)

sample_markdown = """
# LangChain Tutorial

## Installation
To install LangChain, use pip:
```bash
pip install langchain
```

## Basic Usage
Here's a simple example:
```python
from langchain.llms import OpenAI
llm = OpenAI()
```

## Advanced Features
LangChain supports many advanced features like chains and agents.
"""

md_chunks = markdown_splitter.split_text(sample_markdown)
print(f"Markdown split into {len(md_chunks)} chunks")

with open('/root/document-loader.txt', 'w') as f:
    f.write("DOCUMENT_LOADER_COMPLETE")