from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["product", "feature"],
    template="Generate a marketing slogan for {product} highlighting {feature}."
)

prompt = template.format(product="LangChain", feature="AI orchestration")
print("Generated prompt:", prompt)

examples = [
    {"product": "Smartphone", "feature": "camera quality"},
    {"product": "Electric Car", "feature": "eco-friendly"},
    {"product": "AI Assistant", "feature": "natural conversation"}
]
for example in examples:
    print("•", template.format(**example))

with open('/root/basic-templates.txt', 'w') as f:
    f.write("BASIC_TEMPLATES_COMPLETE")