import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.output_parsers import StrOutputParser
from langchain.memory import ConversationSummaryMemory
from langchain_core.messages import HumanMessage, AIMessage

# Setup model
model = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    temperature=0.7,
    base_url=os.environ.get("OPENAI_API_BASE")
)

# Summary Memory Implementation
print("=== Summary Memory ===")

summary_memory = ConversationSummaryMemory(
    llm=model,
    return_messages=True
)

# Add some conversation history
summary_memory.chat_memory.add_user_message("Hi, I'm working on a machine learning project about predicting house prices")
summary_memory.chat_memory.add_ai_message("That sounds interesting! House price prediction is a classic regression problem. What features are you planning to use?")
summary_memory.chat_memory.add_user_message("I have data on square footage, number of bedrooms, bathrooms, and location")
summary_memory.chat_memory.add_ai_message("Great features! Location is particularly important. Are you considering using linear regression or more advanced algorithms?")

# Get summary
summary = summary_memory.predict_new_summary(
    summary_memory.chat_memory.messages,
    ""
)
print("Conversation Summary:", summary)

# Window Memory Implementation
print("\n=== Window Memory (Last K Messages) ===")

class WindowMemory:
    def __init__(self, k=4):
        self.k = k
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)
        # Keep only last k messages
        if len(self.messages) > self.k:
            self.messages = self.messages[-self.k:]

    def get_messages(self):
        return self.messages

    def clear(self):
        self.messages = []

# Demo window memory
window_memory = WindowMemory(k=4)

# Simulate conversation
conversations = [
    ("Tell me about Python", "Python is a versatile programming language"),
    ("What about machine learning?", "Python has great ML libraries like scikit-learn"),
    ("How about deep learning?", "PyTorch and TensorFlow are popular for deep learning"),
    ("What about data visualization?", "Matplotlib and Seaborn are excellent for visualization"),
    ("Can you remind me what we discussed first?", "I'll focus on our recent conversation about data visualization")
]

for human_msg, ai_msg in conversations:
    window_memory.add_message(HumanMessage(content=human_msg))
    window_memory.add_message(AIMessage(content=ai_msg))

    print(f"Human: {human_msg}")
    print(f"AI: {ai_msg}")
    print(f"Memory size: {len(window_memory.get_messages())}")
    print()

print("Final memory contents (last 4 messages):")
for i, msg in enumerate(window_memory.get_messages()):
    print(f"{i+1}. {msg.type}: {msg.content}")

# Custom Memory Store with Different Strategies
print("\n=== Custom Memory Strategies ===")

class CustomMemoryStore:
    def __init__(self):
        self.sessions = {}

    def get_summary_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "type": "summary",
                "memory": ConversationSummaryMemory(llm=model, return_messages=True)
            }
        return self.sessions[session_id]["memory"]

    def get_window_session(self, session_id, k=6):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "type": "window",
                "memory": WindowMemory(k=k)
            }
        return self.sessions[session_id]["memory"]

# Demo different memory types
custom_store = CustomMemoryStore()

# Summary memory for long conversations
summary_mem = custom_store.get_summary_session("long_session")
print("Summary memory created for long conversations")

# Window memory for short conversations
window_mem = custom_store.get_window_session("short_session", k=4)
print("Window memory created for short conversations")

with open('/root/advanced-memory.txt', 'w') as f:
    f.write("ADVANCED_MEMORY_COMPLETE")