import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Setup model
model = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    temperature=0,
    base_url=os.environ.get("OPENAI_API_BASE")
)

# Basic chain
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} briefly"
)

chain = prompt | model | StrOutputParser()

# Streaming demonstration
print("=== Streaming ===")
for chunk in chain.stream({"topic": "quantum computing"}):
    print(chunk, end="", flush=True)
print("\n")

# Batch processing
print("=== Batch Processing ===")
topics = [
    {"topic": "machine learning"},
    {"topic": "blockchain"},
    {"topic": "cloud computing"}
]

batch_results = chain.batch(topics)
for i, result in enumerate(batch_results):
    print(f"Topic {i+1}: {result[:100]}...")

# Async processing
async def async_demo():
    print("\n=== Async Processing ===")
    result = await chain.ainvoke({"topic": "artificial intelligence"})
    print("Async result:", result[:100] + "...")

# Run async demo
asyncio.run(async_demo())

# Fallback chains
print("\n=== Fallback Chains ===")
primary_model = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    temperature=0,
    base_url=os.environ.get("OPENAI_API_BASE")
)

backup_model = ChatOpenAI(
    model="deepseek/deepseek-chat",
    temperature=0,
    base_url=os.environ.get("OPENAI_API_BASE")
)

# Create fallback chain
fallback_chain = (prompt | primary_model | StrOutputParser()).with_fallbacks(
    [prompt | backup_model | StrOutputParser()]
)

result = fallback_chain.invoke({"topic": "neural networks"})
print("Fallback result:", result[:100] + "...")

# Bind additional parameters
print("\n=== Bind Parameters ===")
bound_model = model.bind(max_tokens=50)
bound_chain = prompt | bound_model | StrOutputParser()

result = bound_chain.invoke({"topic": "deep learning"})
print("Bound result:", result)

# Map over inputs
print("\n=== Map Processing ===")
def process_topic(topic_dict):
    topic = topic_dict["topic"]
    return f"Learning about: {topic}"

map_chain = RunnableLambda(process_topic)
mapped_results = map_chain.map().invoke(topics)
for result in mapped_results:
    print(result)

with open('/root/advanced-lcel.txt', 'w') as f:
    f.write("ADVANCED_LCEL_COMPLETE")