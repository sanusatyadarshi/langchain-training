import os
from langchain_openai import ChatOpenAI

# Get configuration from environment variables (strip quotes)
api_base = os.environ.get("OPENAI_API_BASE", "").strip().strip('"').strip("'")
api_key = os.environ.get("OPENAI_API_KEY", "").strip().strip('"').strip("'")
default_model = os.environ.get("DEFAULT_MODEL", "gpt-4")
use_real_api = os.environ.get("USE_REAL_API", "false").lower() == "true"

print(f"🔧 Configuration:")
print(f"   Model: {default_model}")
print(f"   API Base: {api_base if api_base else 'Not configured'}")
print(f"   Real API: {'Enabled' if use_real_api and api_key else 'Demo mode'}")
print()

if use_real_api and api_key and api_base:
    print("🚀 Using real AI model...")
    model = ChatOpenAI(
        model=default_model,
        temperature=0,
        openai_api_key=api_key,
        openai_api_base=api_base
    )

    try:
        response1 = model.invoke("Hello! What's 2+2?")
        print("Q: Hello! What's 2+2?")
        print(f"A: {response1.content}")
        print()

        response2 = model.invoke("What's the capital of France?")
        print("Q: What's the capital of France?")
        print(f"A: {response2.content}")

    except Exception as e:
        print(f"❌ API Error: {e}")
        print("💡 Check your .env file configuration")
        use_real_api = False

if not use_real_api or not api_key or not api_base:
    print("📚 Demo mode - showing example responses:")
    print("Q: Hello! What's 2+2?")
    print("A: Hello! 2+2 equals 4. This is a basic arithmetic calculation.")
    print()
    print("Q: What's the capital of France?")
    print("A: The capital of France is Paris. It's known for landmarks like the Eiffel Tower and Louvre Museum.")

with open('/root/first-model.txt', 'w') as f:
    f.write("FIRST_MODEL_COMPLETE")