"""
LangChain Workshop Configuration Utility
Provides consistent model configuration across all workshop tasks.
"""

import os
from langchain_openai import ChatOpenAI
from typing import Optional

class WorkshopConfig:
    """Centralized configuration for the LangChain workshop."""

    def __init__(self):
        # API Configuration (strip quotes from environment variables)
        self.api_base = os.environ.get("OPENAI_API_BASE", "").strip().strip('"').strip("'")
        self.api_key = os.environ.get("OPENAI_API_KEY", "").strip().strip('"').strip("'")
        self.use_real_api = os.environ.get("USE_REAL_API", "false").lower() == "true"

        # Model Configuration
        self.default_model = os.environ.get("DEFAULT_MODEL", "gpt-4")
        self.fast_model = os.environ.get("FAST_MODEL", "gpt-3.5-turbo")
        self.coding_model = os.environ.get("CODING_MODEL", "gpt-4")
        self.creative_model = os.environ.get("CREATIVE_MODEL", "gpt-4")

        # Other Configuration
        self.api_timeout = int(os.environ.get("API_TIMEOUT", "30"))
        self.max_tokens = int(os.environ.get("MAX_TOKENS", "1000"))
        self.debug_mode = os.environ.get("DEBUG_MODE", "false").lower() == "true"

    @property
    def is_api_configured(self) -> bool:
        """Check if API is properly configured."""
        return bool(self.api_key and self.api_base and self.use_real_api)

    def get_model(self, model_type: str = "default", temperature: float = 0, **kwargs) -> Optional[ChatOpenAI]:
        """
        Get a configured ChatOpenAI model instance.

        Args:
            model_type: Type of model ("default", "fast", "coding", "creative")
            temperature: Temperature setting for the model
            **kwargs: Additional parameters for ChatOpenAI

        Returns:
            ChatOpenAI instance if API is configured, None otherwise
        """
        if not self.is_api_configured:
            return None

        model_names = {
            "default": self.default_model,
            "fast": self.fast_model,
            "coding": self.coding_model,
            "creative": self.creative_model
        }

        model_name = model_names.get(model_type, self.default_model)

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=self.api_key,
            openai_api_base=self.api_base,
            timeout=self.api_timeout,
            max_tokens=self.max_tokens,
            **kwargs
        )

    def print_status(self):
        """Print current configuration status."""
        print(f"🔧 Workshop Configuration:")
        print(f"   API Base: {self.api_base if self.api_base else 'Not configured'}")
        print(f"   API Key: {'***' + self.api_key[-4:] if self.api_key else 'Not configured'}")
        print(f"   Real API: {'Enabled' if self.is_api_configured else 'Demo mode'}")
        print(f"   Default Model: {self.default_model}")
        print(f"   Fast Model: {self.fast_model}")
        print(f"   Coding Model: {self.coding_model}")
        print(f"   Creative Model: {self.creative_model}")
        print()

# Global configuration instance
config = WorkshopConfig()

def demo_response(prompt: str, model_type: str = "default") -> str:
    """
    Generate demo responses when API is not available.

    Args:
        prompt: The input prompt
        model_type: Type of model to simulate

    Returns:
        Demo response string
    """
    demo_responses = {
        "hello": "Hello! I'm a demo response. To get real AI responses, configure your .env file with LiteLLM credentials.",
        "math": "I can help with math! For example, 2+2=4. Configure your .env file to get real AI calculations.",
        "code": """def example_function():
    \"\"\"This is a demo code response.\"\"\"
    return "Configure your .env file for real AI-generated code!"

# Real AI models will provide much better responses!""",
        "creative": "This is a demo creative response! Real AI models can write stories, poems, and much more creative content when you configure your .env file.",
        "default": "This is a demo response. Configure your .env file with your LiteLLM credentials to get real AI responses!"
    }

    # Simple keyword matching for demo responses
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in ["hello", "hi", "greet"]):
        return demo_responses["hello"]
    elif any(word in prompt_lower for word in ["math", "calculate", "+", "-", "*", "/"]):
        return demo_responses["math"]
    elif any(word in prompt_lower for word in ["code", "function", "python", "program"]):
        return demo_responses["code"]
    elif any(word in prompt_lower for word in ["story", "poem", "creative", "write"]):
        return demo_responses["creative"]
    else:
        return demo_responses["default"]

def safe_invoke(model: Optional[ChatOpenAI], prompt: str, model_type: str = "default") -> str:
    """
    Safely invoke a model with fallback to demo mode.

    Args:
        model: ChatOpenAI model instance (can be None)
        prompt: Input prompt
        model_type: Type of model for demo responses

    Returns:
        Model response or demo response
    """
    if model is None:
        return demo_response(prompt, model_type)

    try:
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"❌ API Error: {e}")
        print("💡 Falling back to demo mode")
        return demo_response(prompt, model_type)