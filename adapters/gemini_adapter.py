import os
from typing import Any, Dict, List, Optional
from adapters.base import BaseLLMAdapter, LLMResponse, Message, Role, ToolCall, ToolSpec


class GeminiAdapter(BaseLLMAdapter):
    """
    Adapter for Google Gemini models using the official `google-genai` SDK.
    """

    def __init__(self, model_name: str = "gemini-3.1-flash-lite", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_name, api_key=api_key or os.getenv("GEMINI_API_KEY"), **kwargs)
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please provide it or set it in your environment.")

        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "Please install the Google GenAI SDK: 'pip install google-genai'"
            )

    def _convert_tools(self, tools: Optional[List[ToolSpec]]):
        if not tools:
            return None

        from google.genai import types

        function_declarations = []
        for t in tools:
            decl = types.FunctionDeclaration(
                name=t.name,
                description=t.description,
                parameters=t.parameters,
            )
            function_declarations.append(decl)

        return [types.Tool(function_declarations=function_declarations)]

    def _convert_messages(self, messages: List[Message]):
        from google.genai import types

        contents = []
        system_instruction = None

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content
                continue

            # If raw Content object from Gemini is already preserved, keep it intact (includes thought_signature)
            if msg.raw_content is not None:
                contents.append(msg.raw_content)
                continue

            if msg.role == Role.USER:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=msg.content or "")],
                    )
                )
            elif msg.role == Role.TOOL and msg.tool_results:
                parts = []
                for tr in msg.tool_results:
                    parts.append(
                        types.Part.from_function_response(
                            name=tr.name,
                            response={"result": tr.output},
                        )
                    )
                contents.append(types.Content(role="user", parts=parts))

        return system_instruction, contents

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSpec]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        from google.genai import types

        system_instruction, contents = self._convert_messages(messages)
        gemini_tools = self._convert_tools(tools)

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            tools=gemini_tools,
        )
        if max_tokens:
            config.max_output_tokens = max_tokens

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        text_content = ""
        tool_calls = []
        raw_content = None

        if response.candidates and response.candidates[0].content:
            raw_content = response.candidates[0].content
            if raw_content.parts:
                for idx, part in enumerate(raw_content.parts):
                    if hasattr(part, "text") and part.text:
                        text_content += part.text
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{idx}_{fc.name}",
                                name=fc.name,
                                arguments=dict(fc.args) if fc.args else {},
                            )
                        )

        return LLMResponse(
            text=text_content if text_content else None,
            tool_calls=tool_calls,
            raw_response=raw_content,
        )
