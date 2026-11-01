import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """Standardized representation of an LLM requesting a tool execution."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Standardized result returned from a tool back to the LLM."""
    call_id: str
    name: str
    output: str
    is_error: bool = False


@dataclass
class Message:
    """Standardized chat message across all LLM providers."""
    role: Role
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_results: Optional[List[ToolResult]] = None
    raw_content: Optional[Any] = None


@dataclass
class ToolParameter:
    name: str
    type: str  # "string", "integer", "boolean", "number", "object", "array"
    description: str
    required: bool = True


@dataclass
class ToolSpec:
    """Standardized tool specification schema."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema for arguments
    handler: Optional[Callable[..., Any]] = None

    def execute(self, **kwargs) -> str:
        if not self.handler:
            raise ValueError(f"No execution handler registered for tool '{self.name}'.")
        try:
            result = self.handler(**kwargs)
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False, indent=2)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{self.name}': {str(e)}"


@dataclass
class LLMResponse:
    """Standardized LLM generation response."""
    text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw_response: Optional[Any] = None
    usage: Dict[str, int] = field(default_factory=dict)


class BaseLLMAdapter(ABC):
    """Abstract Base Class (Interface) for all LLM providers."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.kwargs = kwargs

    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Takes standardized messages & tools, formats them to the provider SDK,
        calls the LLM, and converts response to standardized LLMResponse.
        """
        raise NotImplementedError

