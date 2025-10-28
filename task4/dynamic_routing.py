import os
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

# Transform functions
def uppercase_transform(text):
    return text.upper()

def add_prefix(text):
    return f"IMPORTANT: {text}"

def route_by_length(text):
    if len(text) < 50:
        return "short"
    elif len(text) < 200:
        return "medium"
    else:
        return "long"

def route_by_type(input_dict):
    text = input_dict["text"]
    if "?" in text:
        return "question"
    elif "!" in text:
        return "exclamation"
    else:
        return "statement"

# Transform demonstrations
print("=== Transform Functions ===")
transform_chain = (
    RunnableLambda(uppercase_transform) |
    RunnableLambda(add_prefix)
)

result = transform_chain.invoke("hello world")
print("Transformed:", result)

# Routing by length
print("\n=== Routing by Length ===")
short_prompt = PromptTemplate(
    input_variables=["text"],
    template="Expand this short text: {text}"
)

medium_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize this medium text: {text}"
)

long_prompt = PromptTemplate(
    input_variables=["text"],
    template="Extract key points from this long text: {text}"
)

def length_router(input_dict):
    text = input_dict["text"]
    length_category = route_by_length(text)

    if length_category == "short":
        return short_prompt | model | StrOutputParser()
    elif length_category == "medium":
        return medium_prompt | model | StrOutputParser()
    else:
        return long_prompt | model | StrOutputParser()

# Test routing
texts = [
    "AI is cool",  # short
    "Artificial intelligence is transforming how we work and live in many different ways",  # medium
    "Artificial intelligence represents one of the most significant technological advances of our time, fundamentally changing industries from healthcare to finance, transportation to education, and creating new opportunities while also presenting challenges that society must carefully navigate"  # long
]

for text in texts:
    route_chain = RunnableLambda(length_router)
    result = route_chain.invoke({"text": text})
    print(f"Text ({len(text)} chars): {text}")
    print(f"Result: {result}\n")

# Multi-lambda chain
print("=== Multi-Lambda Chain ===")
def extract_words(text):
    return {"words": text.split(), "count": len(text.split())}

def filter_long_words(data):
    long_words = [word for word in data["words"] if len(word) > 5]
    return {"long_words": long_words, "original_count": data["count"]}

multi_lambda_chain = (
    RunnableLambda(extract_words) |
    RunnableLambda(filter_long_words)
)

result = multi_lambda_chain.invoke("The sophisticated algorithm processes information efficiently")
print("Multi-lambda result:", result)

with open('/root/dynamic-routing.txt', 'w') as f:
    f.write("DYNAMIC_ROUTING_COMPLETE")