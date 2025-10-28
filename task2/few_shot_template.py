from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "fast", "output": "slow"},
    {"input": "hot", "output": "cold"}
]

example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Find the opposite of each word:",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)

test_words = ["big", "light", "expensive", "difficult"]
for word in test_words:
    print(f"Prompt for {word}:\n", few_shot_template.format(word=word))

selected_examples = examples[:2]
dynamic_template = FewShotPromptTemplate(
    examples=selected_examples,
    example_prompt=example_template,
    prefix="Learn the pattern from these examples:",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)
print(dynamic_template.format(word="bright"))

with open('/root/few-shot-templates.txt', 'w') as f:
    f.write("FEW_SHOT_TEMPLATES_COMPLETE")