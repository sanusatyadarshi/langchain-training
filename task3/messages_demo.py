import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

model = ChatOpenAI(model="openai/gpt-4.1-mini", temperature=0, base_url=os.environ.get("OPENAI_API_BASE"))

messages = [
    SystemMessage(content="You are a helpful Python tutor"),
    HumanMessage(content="Explain variables to a beginner")
]

print(model.invoke(messages).content)