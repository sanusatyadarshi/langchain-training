from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

safe_template = PromptTemplate(
    input_variables=["topic"],
    template="Write a summary about {topic}",
    validate_template=True
)

partial_template = PromptTemplate(
    input_variables=["task"],
    template="As an expert {role}, please help with: {task}",
    partial_variables={"role": "AI consultant"}
)

class ProductReview(BaseModel):
    rating: int = Field(description="Rating from 1-5")
    pros: list[str]
    cons: list[str]
    recommendation: str

parser = PydanticOutputParser(pydantic_object=ProductReview)

structured_template = PromptTemplate(
    template="Review this product: {product}\n{format_instructions}",
    input_variables=["product"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

def create_conditional_template(user_level):
    if user_level == "beginner":
        t = "Explain {concept} in very simple terms with examples."
    elif user_level == "intermediate":
        t = "Explain {concept} with some technical details."
    else:
        t = "Provide a comprehensive technical explanation of {concept}."
    return PromptTemplate(input_variables=["concept"], template=t)

# Demo examples
print("Safe template example:")
print(safe_template.format(topic="machine learning"))

print("\nPartial template example:")
print(partial_template.format(task="optimize database queries"))

print("\nStructured template example:")
print(structured_template.format(product="iPhone 15"))

print("\nConditional template examples:")
for level in ["beginner", "intermediate", "expert"]:
    template = create_conditional_template(level)
    print(f"{level.capitalize()}:", template.format(concept="neural networks"))

with open('/root/advanced-templates.txt', 'w') as f:
    f.write("ADVANCED_TEMPLATES_COMPLETE")