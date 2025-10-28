import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Setup model
model = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    temperature=0,
    base_url=os.environ.get("OPENAI_API_BASE")
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create comprehensive knowledge base
knowledge_docs = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models. It enables applications that are data-aware and agentic.",
        metadata={"source": "langchain_intro", "topic": "framework"}
    ),
    Document(
        page_content="LCEL (LangChain Expression Language) is a declarative way to compose chains. It supports streaming, async, parallel execution, and fallbacks.",
        metadata={"source": "lcel_guide", "topic": "syntax"}
    ),
    Document(
        page_content="Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. It allows LLMs to access external knowledge.",
        metadata={"source": "rag_explained", "topic": "technique"}
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching high-dimensional vectors. They enable semantic similarity search.",
        metadata={"source": "vector_stores", "topic": "storage"}
    ),
    Document(
        page_content="Prompt templates allow you to create reusable, parameterized prompts. They support variables, conditional logic, and output formatting.",
        metadata={"source": "prompt_templates", "topic": "prompts"}
    ),
    Document(
        page_content="Memory systems in LangChain include conversation buffer memory, summary memory, and entity memory for maintaining context.",
        metadata={"source": "memory_systems", "topic": "memory"}
    ),
    Document(
        page_content="LangChain agents can use tools to interact with external APIs, databases, and other systems. They support ReAct and plan-and-execute patterns.",
        metadata={"source": "agents_guide", "topic": "agents"}
    ),
    Document(
        page_content="Document loaders in LangChain support various formats including PDF, CSV, HTML, and text files. They can also load from URLs and databases.",
        metadata={"source": "document_loaders", "topic": "ingestion"}
    ),
    Document(
        page_content="Text splitters break large documents into chunks. RecursiveCharacterTextSplitter is most common, but specialized splitters exist for code and markdown.",
        metadata={"source": "text_splitters", "topic": "preprocessing"}
    ),
    Document(
        page_content="Output parsers structure LLM responses into specific formats like JSON, lists, or custom objects using Pydantic models.",
        metadata={"source": "output_parsers", "topic": "parsing"}
    )
]

print("=== Creating Knowledge Base ===")
vector_store = FAISS.from_documents(knowledge_docs, embeddings)
print(f"Knowledge base created with {len(knowledge_docs)} documents")

# Create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Test retrieval
print("\n=== Testing Retrieval ===")
test_query = "What is LCEL?"
retrieved_docs = retriever.get_relevant_documents(test_query)

print(f"Query: {test_query}")
print(f"Retrieved {len(retrieved_docs)} documents:")
for i, doc in enumerate(retrieved_docs):
    print(f"{i+1}. {doc.page_content}")

# Custom prompt template for RAG
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following context to answer the question. If you cannot answer based on the context, say "I don't have enough information in the provided context."

Context:
{context}

Question: {question}

Answer:"""
)

# Create RAG chain using modern LCEL
print("\n=== Creating RetrievalQA Chain ===")

def format_docs(docs):
    """Format retrieved documents for the prompt"""
    return "\n\n".join(doc.page_content for doc in docs)

# Build the LCEL chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | custom_prompt
    | model
    | StrOutputParser()
)

# Wrapper class to maintain RetrievalQA interface
class RetrievalQAWrapper:
    """Wrapper to maintain compatibility with old RetrievalQA interface"""
    def __init__(self, chain, retriever):
        self.chain = chain
        self.retriever = retriever

    def __call__(self, inputs):
        """Execute chain and return result with source documents"""
        query = inputs.get("query")
        result = self.chain.invoke(query)
        source_docs = self.retriever.get_relevant_documents(query)
        return {
            "result": result,
            "source_documents": source_docs
        }

qa_chain = RetrievalQAWrapper(rag_chain, retriever)

# Test questions
test_questions = [
    "What is LCEL and how does it work?",
    "Explain Retrieval-Augmented Generation",
    "What are the benefits of using vector stores?",
    "How do memory systems work in LangChain?",
    "What types of document loaders are available?",
    "How can I create reusable prompts?",
    "What is the difference between a chain and an agent?",
    "How do I split large documents for processing?"
]

print("\n=== Testing RAG System ===")
for question in test_questions:
    print(f"\nQ: {question}")

    result = qa_chain({"query": question})

    print(f"A: {result['result']}")

    # Show sources
    if result.get('source_documents'):
        print("Sources:")
        for i, doc in enumerate(result['source_documents']):
            source = doc.metadata.get('source', 'unknown')
            topic = doc.metadata.get('topic', 'general')
            print(f"  {i+1}. {source} ({topic})")

# Advanced retrieval with different strategies
print("\n=== Advanced Retrieval Strategies ===")

# MMR retriever (Maximum Marginal Relevance)
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 5}
)

print("MMR Retrieval for 'LangChain components':")
mmr_docs = mmr_retriever.get_relevant_documents("LangChain components")
for i, doc in enumerate(mmr_docs):
    print(f"{i+1}. {doc.page_content[:100]}...")

# Similarity threshold retriever
threshold_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7}
)

print("\nThreshold Retrieval for 'machine learning':")
threshold_docs = threshold_retriever.get_relevant_documents("machine learning")
print(f"Found {len(threshold_docs)} documents above threshold")

# Filter-based retrieval
filter_retriever = vector_store.as_retriever(
    search_kwargs={"k": 3, "filter": {"topic": "memory"}}
)

print("\nFiltered Retrieval for memory-related content:")
filtered_docs = filter_retriever.get_relevant_documents("conversation context")
for doc in filtered_docs:
    print(f"- {doc.page_content}")

with open('/root/retrieval-chain.txt', 'w') as f:
    f.write("RETRIEVAL_CHAIN_COMPLETE")