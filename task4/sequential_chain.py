import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from workshop_config import config, safe_invoke

# Setup model using workshop configuration
model = config.get_model("default", temperature=0)
print(f"🔧 LCEL Configuration: {'Real API' if config.is_api_configured else 'Demo mode'}")

# Basic chain: prompt | model | parser
print("=== Basic Sequential Chain ===")
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a brief explanation about {topic}"
)

if model:
    chain = prompt | model | StrOutputParser()
    result = chain.invoke({"topic": "machine learning"})
    print(result)
else:
    print("Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for each task.")

# Multi-step chain
print("\n=== Multi-step Chain ===")
step1_prompt = PromptTemplate(
    input_variables=["concept"],
    template="Define {concept} in one sentence"
)

step2_prompt = PromptTemplate(
    input_variables=["definition"],
    template="Give a real-world example of: {definition}"
)

# Create multi-step chain
step1_chain = step1_prompt | model | StrOutputParser()
step2_chain = step2_prompt | model | StrOutputParser()

# Execute steps
definition = step1_chain.invoke({"concept": "neural networks"})
print("Definition:", definition)

example = step2_chain.invoke({"definition": definition})
print("Example:", example)

# Passthrough demonstration
print("\n=== Passthrough Chain ===")
passthrough_chain = RunnableParallel({
    "original": RunnablePassthrough(),
    "processed": prompt | model | StrOutputParser()
})

result = passthrough_chain.invoke({"topic": "blockchain"})
print("Original input:", result["original"])
print("Processed output:", result["processed"])

with open('/root/sequential-chain.txt', 'w') as f:
    f.write("SEQUENTIAL_CHAIN_COMPLETE")