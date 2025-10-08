import os
from langchain_openai import ChatOpenAI

# Get configuration from environment variables (strip quotes)
api_base = os.environ.get("OPENAI_API_BASE", "").strip().strip('"').strip("'")
api_key = os.environ.get("OPENAI_API_KEY", "").strip().strip('"').strip("'")
use_real_api = os.environ.get("USE_REAL_API", "false").lower() == "true"

# Model configuration from environment
fast_model_name = os.environ.get("FAST_MODEL", "gpt-3.5-turbo")
coding_model_name = os.environ.get("CODING_MODEL", "gpt-4")
creative_model_name = os.environ.get("CREATIVE_MODEL", "gpt-4")

print(f"🔧 Model Configuration:")
print(f"   Fast Model: {fast_model_name}")
print(f"   Coding Model: {coding_model_name}")
print(f"   Creative Model: {creative_model_name}")
print(f"   Real API: {'Enabled' if use_real_api and api_key else 'Demo mode'}")
print()

if use_real_api and api_key and api_base:
    # Different models for different purposes
    fast_model = ChatOpenAI(
        model=fast_model_name,
        temperature=0,
        openai_api_key=api_key,
        openai_api_base=api_base
    )

    coding_model = ChatOpenAI(
        model=coding_model_name,
        temperature=0,
        openai_api_key=api_key,
        openai_api_base=api_base
    )

    chat_model = ChatOpenAI(
        model=creative_model_name,
        temperature=0.7,
        openai_api_key=api_key,
        openai_api_base=api_base
    )

# Test different models
question = "Write a Python function to calculate fibonacci numbers"

if use_real_api and api_key and api_base:
    try:
        print("=== Fast Model Response ===")
        response1 = fast_model.invoke(question)
        print(response1.content)

        print("\n=== Coding Model Response ===")
        response2 = coding_model.invoke(question)
        print(response2.content)

        print("\n=== Creative Model Response ===")
        response3 = chat_model.invoke("Tell me a fun fact about programming")
        print(response3.content)

    except Exception as e:
        print(f"❌ API Error: {e}")
        print("💡 Falling back to demo mode")
        use_real_api = False

if not use_real_api or not api_key or not api_base:
    print("📚 Demo mode - showing example responses:")
    print("\n=== Fast Model Response ===")
    print("""def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage:
print(fibonacci(10))  # Output: 55""")

    print("\n=== Coding Model Response ===")
    print("""def fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number using dynamic programming.\"\"\"
    if n <= 1:
        return n

    # Use iterative approach for better performance
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Example:
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")""")

    print("\n=== Creative Model Response ===")
    print("Fun fact: The term 'bug' in programming originated in 1947 when Admiral Grace Hopper found an actual moth stuck in a computer relay at Harvard University. She taped the bug to her logbook and coined the term 'debugging'!")

with open('/root/multiple-models.txt', 'w') as f:
    f.write("MULTIPLE_MODELS_COMPLETE")