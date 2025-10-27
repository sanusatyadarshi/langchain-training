
#############################################################################################

# Using LangChain, we can easily change providers thus completely avoiding the vendor lock-in

#############################################################################################

################################################### OPENAI ###################################
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

################################################### ANTHROPIC ################################


from langchain_openai import ChatAnthropic

llm = ChatAnthropic(model="claude-3-sonnet...")