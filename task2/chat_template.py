from langchain.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} expert with {years} years of experience."),
    ("human", "Explain {concept} to me in simple terms."),
    ("assistant", "I'll explain {concept} step by step.")
])

messages = chat_template.format_messages(
    role="Python programming", years="10", concept="decorators"
)
for msg in messages:
    print(msg.type.upper() + ":", msg.content)

scenarios = [
    {"role": "Data Science", "years": "5", "concept": "machine learning"},
    {"role": "Web Development", "years": "8", "concept": "REST APIs"},
]
for scenario in scenarios:
    messages = chat_template.format_messages(**scenario)
    print("System:", messages[0].content)

with open('/root/chat-templates.txt', 'w') as f:
    f.write("CHAT_TEMPLATES_COMPLETE")