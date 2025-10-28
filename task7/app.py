import os
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Initialize model
model = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    temperature=0.7,
    base_url=os.environ.get("OPENAI_API_BASE")
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create knowledge base
knowledge_docs = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models. It enables applications that are data-aware and agentic, allowing language models to connect with other sources of data and interact with their environment.",
        metadata={"source": "langchain_intro", "topic": "framework"}
    ),
    Document(
        page_content="LCEL (LangChain Expression Language) is a declarative way to compose chains. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest prompt + LLM chain to the most complex chains. It supports streaming, async, parallel execution, retries and fallbacks.",
        metadata={"source": "lcel_guide", "topic": "syntax"}
    ),
    Document(
        page_content="Retrieval-Augmented Generation (RAG) is a technique that combines retrieval of relevant documents with generation of responses. This allows language models to access external knowledge beyond their training data, making them more accurate and up-to-date.",
        metadata={"source": "rag_explained", "topic": "technique"}
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching high-dimensional vectors. In the context of RAG, they store embeddings of documents that can be efficiently searched for semantic similarity. Popular vector stores include FAISS, Pinecone, and Chroma.",
        metadata={"source": "vector_stores", "topic": "storage"}
    ),
    Document(
        page_content="Prompt templates are a powerful way to create reusable prompts for language models. They allow you to parameterize prompts and create structured inputs for consistent model behavior. LangChain supports basic templates, chat templates, and few-shot templates.",
        metadata={"source": "prompt_templates", "topic": "prompts"}
    ),
    Document(
        page_content="Memory systems in LangChain allow chatbots to maintain conversation context. Types include ConversationBufferMemory for storing all messages, ConversationSummaryMemory for summarizing old messages, and ConversationBufferWindowMemory for keeping only recent messages.",
        metadata={"source": "memory_systems", "topic": "memory"}
    ),
    Document(
        page_content="LangChain agents are systems that can use tools to interact with external systems and APIs. They can reason about which tools to use and in what order. Common agent types include ReAct agents and plan-and-execute agents.",
        metadata={"source": "agents_guide", "topic": "agents"}
    ),
    Document(
        page_content="Document loaders in LangChain support ingesting text from various sources including PDF files, CSV files, HTML pages, and plain text files. They can also load content from URLs, databases, and APIs. Each loader is optimized for its specific format.",
        metadata={"source": "document_loaders", "topic": "ingestion"}
    )
]

print("Initializing LangChain AI Assistant...")
print("Creating vector store...")

# Create vector store
vector_store = FAISS.from_documents(knowledge_docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Create RAG chain
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful LangChain expert. Use the following context to answer the question accurately. If you cannot answer based on the context, use your general knowledge but mention that the information is not from the provided context.

Context:
{context}

Question: {question}

Answer:"""
)

def format_docs(docs):
    """Format retrieved documents for the prompt"""
    return "\n\n".join(doc.page_content for doc in docs)

# Build the LCEL RAG chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
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

# Memory setup for conversation
memory_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in memory_store:
        memory_store[session_id] = ChatMessageHistory()
    return memory_store[session_id]

# Chat chain with memory
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful LangChain expert assistant. Be conversational and helpful."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chat_chain = chat_prompt | model | StrOutputParser()

memory_chat_chain = RunnableWithMessageHistory(
    chat_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

def chat_with_rag(message, history, use_rag=True):
    """Main chat function that can use RAG or regular conversation"""
    try:
        if use_rag:
            # Use RAG for knowledge-based questions
            result = qa_chain({"query": message})
            response = result['result']

            # Add source information
            if result.get('source_documents'):
                sources = [doc.metadata.get('source', 'unknown') for doc in result['source_documents']]
                unique_sources = list(set(sources))
                if unique_sources:
                    response += f"\n\n*Sources: {', '.join(unique_sources)}*"
        else:
            # Use conversational chain with memory
            session_config = {"configurable": {"session_id": "default_session"}}
            response = memory_chat_chain.invoke(
                {"input": message},
                config=session_config
            )

    except Exception as e:
        response = f"I apologize, but I encountered an error: {str(e)}. Please try again."

    return response

def gradio_chat(message, history, use_rag):
    """Gradio-compatible chat function"""
    response = chat_with_rag(message, history, use_rag)

    # Format for Gradio
    history.append([message, response])
    return "", history

# Sample prompts for testing
sample_prompts = [
    "What is LCEL and how does it work?",
    "Explain Retrieval-Augmented Generation",
    "How do I create prompt templates?",
    "What are the different types of memory in LangChain?",
    "How do vector stores work?",
    "What's the difference between chains and agents?",
    "How can I load documents from different sources?",
    "Tell me about LangChain's main components"
]

# Create Gradio interface
with gr.Blocks(title="LangChain AI Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 LangChain AI Assistant")
    gr.Markdown("Ask questions about LangChain! Toggle RAG mode to switch between knowledge-based and conversational responses.")

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=400, label="Chat")

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask me about LangChain...",
                    label="Your Message",
                    scale=4
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

            use_rag = gr.Checkbox(
                label="Use RAG (Knowledge Base)",
                value=True,
                info="Toggle to switch between RAG and conversational mode"
            )

            clear = gr.Button("Clear Chat", variant="secondary")

        with gr.Column(scale=1):
            gr.Markdown("### 💡 Sample Questions")

            def create_sample_button(prompt):
                return gr.Button(prompt, size="sm")

            for prompt in sample_prompts:
                btn = create_sample_button(prompt)
                btn.click(
                    lambda p=prompt: p,
                    outputs=msg
                )

    # Event handlers
    def submit_message(message, history, rag_mode):
        if message.strip():
            return gradio_chat(message, history, rag_mode)
        return message, history

    send_btn.click(
        submit_message,
        inputs=[msg, chatbot, use_rag],
        outputs=[msg, chatbot]
    )

    msg.submit(
        submit_message,
        inputs=[msg, chatbot, use_rag],
        outputs=[msg, chatbot]
    )

    clear.click(lambda: ([], {}), outputs=[chatbot], show_progress=False)

print("LangChain AI Assistant ready!")
print("Starting Gradio interface...")

if __name__ == "__main__":
    # Create checkpoint file
    with open('/root/langchain-chatbot-ready.txt', 'w') as f:
        f.write("LANGCHAIN_CHATBOT_READY")

    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )