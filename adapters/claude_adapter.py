import os
from typing import Any, Dict, List, Optional
from adapters.base import BaseLLMAdapter, LLMResponse, Message, Role, ToolCall, ToolSpec


class ClaudeAdapter(BaseLLMAdapter):
    """
    Adapter for Anthropic Claude models using the official `anthropic` SDK.
    """

    def __init__(self, model_name: str = "claude-3-7-sonnet-20250219", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_name, api_key=api_key or os.getenv("ANTHROPIC_API_KEY"), **kwargs)
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set. Please provide it or set it in your environment.")

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "Please install the Anthropic SDK: 'pip install anthropic'"
            )

    def _convert_tools(self, tools: Optional[List[ToolSpec]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None

        # Anthropic format: name, description, input_schema
        claude_tools = []
        for t in tools:
            claude_tools.append({
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            })
        return claude_tools

    def _convert_messages(self, messages: List[Message]):
        system_prompt = ""
        claude_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_prompt = msg.content
                continue

            if msg.role == Role.USER:
                claude_messages.append({
                    "role": "user",
                    "content": msg.content,
                })
            elif msg.role == Role.ASSISTANT:
                content_blocks = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                claude_messages.append({
                    "role": "assistant",
                    "content": content_blocks,
                })
            elif msg.role == Role.TOOL:
                content_blocks = []
                if msg.tool_results:
                    for tr in msg.tool_results:
                        content_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tr.call_id,
                            "content": tr.output,
                            "is_error": tr.is_error,
                        })
                claude_messages.append({
                    "role": "user",
                    "content": content_blocks,
                })

        return system_prompt, claude_messages

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = 4096,
    ) -> LLMResponse:
        system_prompt, claude_messages = self._convert_messages(messages)
        claude_tools = self._convert_tools(tools)

        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": claude_messages,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
        }

        if system_prompt:
            params["system"] = system_prompt

        if claude_tools:
            params["tools"] = claude_tools

        response = self.client.messages.create(**params)

        text_content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        return LLMResponse(
            text=text_content if text_content else None,
            tool_calls=tool_calls,
            raw_response=response,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

