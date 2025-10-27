from openai import OpenAI

client = OpenAI(api_keys="your_key_here")

response = client.chat.completion.create(
    model = "gpt-4o-mini",
    messages = [
        {"role": "system", "content": "You are company chatbot"},
        {"role": "user", "content": "What does our company policy say?" }
    ]
)

print(response.choice[0].message["content"])