from typing import Optional
from adapters.base import BaseLLMAdapter
from adapters.gemini_adapter import GeminiAdapter
from adapters.claude_adapter import ClaudeAdapter


def get_llm_adapter(
    model_name: str,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseLLMAdapter:
    """
    Factory function to instantiate the correct LLM adapter based on model name.
    
    Examples:
        adapter = get_llm_adapter("gemini-2.5-flash")
        adapter = get_llm_adapter("claude-3-7-sonnet-20250219")
    """
    normalized = model_name.lower()

    if "gemini" in normalized:
        return GeminiAdapter(model_name=model_name, api_key=api_key, **kwargs)
    elif "claude" in normalized or "anthropic" in normalized:
        return ClaudeAdapter(model_name=model_name, api_key=api_key, **kwargs)
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Currently supported: Gemini, Claude."
        )

