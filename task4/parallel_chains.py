import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Setup model
model = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    temperature=0.7,
    base_url=os.environ.get("OPENAI_API_BASE")
)

# Create different prompts
joke_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell a joke about {topic}"
)

fact_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Share an interesting fact about {topic}"
)

poem_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short poem about {topic}"
)

# Create parallel chains
print("=== Parallel Execution ===")
parallel_chain = RunnableParallel(
    joke=joke_prompt | model | StrOutputParser(),
    fact=fact_prompt | model | StrOutputParser(),
    poem=poem_prompt | model | StrOutputParser()
)

results = parallel_chain.invoke({"topic": "programming"})
print("Joke:", results["joke"])
print("Fact:", results["fact"])
print("Poem:", results["poem"])

# Analysis pipeline
print("\n=== Analysis Pipeline ===")
analysis_chain = RunnableParallel(
    sentiment=PromptTemplate(
        input_variables=["text"],
        template="Analyze the sentiment of: {text}"
    ) | model | StrOutputParser(),

    summary=PromptTemplate(
        input_variables=["text"],
        template="Summarize in one sentence: {text}"
    ) | model | StrOutputParser(),

    keywords=PromptTemplate(
        input_variables=["text"],
        template="Extract 3 keywords from: {text}"
    ) | model | StrOutputParser()
)

text_to_analyze = "I love working with Python! It makes programming so enjoyable and productive."
analysis_results = analysis_chain.invoke({"text": text_to_analyze})

print("Text:", text_to_analyze)
print("Sentiment:", analysis_results["sentiment"])
print("Summary:", analysis_results["summary"])
print("Keywords:", analysis_results["keywords"])

with open('/root/parallel-chains.txt', 'w') as f:
    f.write("PARALLEL_CHAINS_COMPLETE")