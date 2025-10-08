import os
from langchain_openai import ChatOpenAI

# Temperature demonstration
print("=== Temperature Effects ===")
creative_model = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    temperature=0.9,
    base_url=os.environ.get("OPENAI_API_BASE")
)

precise_model = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    temperature=0,
    base_url=os.environ.get("OPENAI_API_BASE")
)

prompt = "Write a short poem about coding"
print("Creative (temp=0.9):", creative_model.invoke(prompt).content)
print("Precise (temp=0):", precise_model.invoke(prompt).content)

# Max tokens demonstration
print("\n=== Max Tokens Control ===")
limited_model = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    temperature=0,
    max_tokens=50,
    base_url=os.environ.get("OPENAI_API_BASE")
)

print("Limited tokens:", limited_model.invoke("Explain machine learning").content)

# Streaming demonstration
print("\n=== Streaming Response ===")
streaming_model = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    temperature=0,
    streaming=True,
    base_url=os.environ.get("OPENAI_API_BASE")
)

print("Streaming response:")
for chunk in streaming_model.stream("What are the benefits of Python?"):
    print(chunk.content, end="", flush=True)
print()  # New line after streaming