import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from workshop_config import config

# Setup model using workshop configuration
if config.is_api_configured:
    model = config.get_model("default", temperature=0.7)
    print("🚀 Using real AI model for memory demonstration...")
else:
    print("📚 Demo mode: Memory concepts will be demonstrated without API calls")
    model = None

# Create prompt with memory placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who remembers our conversation."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create chain
chain = prompt | model | StrOutputParser()

# Create message history store
store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Create memory-enabled chain
memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Demo conversation
print("=== Memory Demonstration ===")
session_config = {"configurable": {"session_id": "user123"}}

if model:
    try:
        # First interaction
        response1 = memory_chain.invoke(
            {"input": "Hi, my name is Alice and I'm a data scientist"},
            config=session_config
        )
        print("Assistant:", response1)

        # Second interaction - should remember
        response2 = memory_chain.invoke(
            {"input": "What's my name and profession?"},
            config=session_config
        )
        print("Assistant:", response2)

        # Third interaction - continuing the conversation
        response3 = memory_chain.invoke(
            {"input": "Can you help me with a Python machine learning project?"},
            config=session_config
        )
        print("Assistant:", response3)

    except Exception as e:
        print(f"❌ API Error: {e}")
        print("💡 Falling back to demo mode")
        model = None

if not model:
    print("📚 Demo mode - showing memory concept:")
    print("User: Hi, my name is Alice and I'm a data scientist")
    print("Assistant: Hello Alice! Nice to meet you. Data science is a fascinating field. How can I help you today?")
    print()
    print("User: What's my name and profession?")
    print("Assistant: Your name is Alice and you're a data scientist. I remember from our conversation just now!")
    print()
    print("User: Can you help me with a Python machine learning project?")
    print("Assistant: Absolutely, Alice! As a data scientist, I'd be happy to help with your Python ML project. What specific aspect are you working on?")

# Show different session
print("\n=== Different Session (No Memory) ===")
new_session_config = {"configurable": {"session_id": "user456"}}

if model:
    response4 = memory_chain.invoke(
        {"input": "What's my name?"},
        config=new_session_config
    )
    print("Assistant:", response4)
else:
    print("Assistant: I don't have any information about your name since this is a new session.")

# Show memory contents
print("\n=== Memory Contents ===")
history = get_session_history("user123")
for message in history.messages:
    print(f"{message.type}: {message.content}")

with open('/root/memory-fundamentals.txt', 'w') as f:
    f.write("MEMORY_FUNDAMENTALS_COMPLETE")