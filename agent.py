from typing import Callable, List, Optional
import json
from adapters.base import BaseLLMAdapter, Message, Role, ToolCall, ToolResult, ToolSpec


DEFAULT_SYSTEM_PROMPT = """You are an autonomous AI coding assistant running inside a Harness environment.
You have access to tools to inspect the environment, read/write files, and execute shell commands.
When solving tasks:
1. Always analyze what needs to be done.
2. Use tools to inspect or modify files as appropriate.
3. Be concise and accurate.
"""


class HarnessAgent:
    """
    Core ReAct loop runner that coordinates the LLM adapter and tool executions.
    """

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        tools: Optional[List[ToolSpec]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 30,
        on_step_callback: Optional[Callable[[str], None]] = None,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ):
        self.adapter = adapter
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.on_step_callback = on_step_callback or (lambda msg: print(msg))
        self.on_event = on_event

        self.tool_map = {t.name: t for t in self.tools}
        self.history: List[Message] = [
            Message(role=Role.SYSTEM, content=self.system_prompt)
        ]

    def save_history(self, file_path: str):
        """Saves self.history to a JSON file."""
        data = []
        for msg in self.history:
            msg_dict = {
                "role": msg.role.value,
                "content": msg.content,
            }
            if msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in msg.tool_calls
                ]
            if msg.tool_results:
                msg_dict["tool_results"] = [
                    {"call_id": tr.call_id, "name": tr.name, "output": tr.output, "is_error": tr.is_error}
                    for tr in msg.tool_results
                ]
            data.append(msg_dict)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_history(self, file_path: str):
        """Loads history from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.history = []
        for item in data:
            tool_calls = None
            if "tool_calls" in item:
                tool_calls = [ToolCall(**tc) for tc in item["tool_calls"]]
            
            tool_results = None
            if "tool_results" in item:
                tool_results = [ToolResult(**tr) for tr in item["tool_results"]]
            
            self.history.append(Message(
                role=Role(item["role"]),
                content=item["content"],
                tool_calls=tool_calls,
                tool_results=tool_results
            ))

    def log(self, text: str):
        if self.on_step_callback:
            self.on_step_callback(text)

    def emit(self, event_type: str, data: dict):
        if self.on_event:
            try:
                self.on_event(event_type, data)
            except Exception:
                pass

    def run(self, user_prompt: str) -> str:
        """
        Executes a prompt through the ReAct loop until the agent completes the task
        or reaches max_steps.
        """
        self.log(f"\n[USER]: {user_prompt}\n" + "-" * 50)
        self.emit("user", {"prompt": user_prompt})
        self.history.append(Message(role=Role.USER, content=user_prompt))

        for step in range(1, self.max_steps + 1):
            self.log(f"\n[STEP {step}/{self.max_steps}] Calling LLM ({self.adapter.model_name})...")
            self.emit("step", {"step": step, "max_steps": self.max_steps, "model": self.adapter.model_name})

            response = self.adapter.generate(
                messages=self.history,
                tools=self.tools,
            )

            # Check if LLM requested tool execution
            if response.tool_calls:
                # 1. Record assistant message with tool calls & raw response
                self.history.append(
                    Message(
                        role=Role.ASSISTANT,
                        content=response.text or "",
                        tool_calls=response.tool_calls,
                        raw_content=response.raw_response,
                    )
                )

                if response.text:
                    self.log(f"[ASSISTANT THOUGHT]: {response.text}")
                    self.emit("thought", {"text": response.text})

                # 2. Execute requested tools
                tool_results: List[ToolResult] = []
                for tc in response.tool_calls:
                    self.log(f"[TOOL CALL] -> {tc.name}({tc.arguments})")
                    self.emit("tool_call", {"id": tc.id, "name": tc.name, "arguments": tc.arguments})
                    tool = self.tool_map.get(tc.name)

                    if not tool:
                        result_str = f"Error: Tool '{tc.name}' not found."
                        is_error = True
                    else:
                        try:
                            result_str = tool.execute(**tc.arguments)
                            is_error = False
                        except Exception as e:
                            result_str = f"Error executing tool '{tc.name}': {str(e)}"
                            is_error = True

                    preview = result_str[:200] + ("..." if len(result_str) > 200 else "")
                    self.log(f"[TOOL RESULT] <- {preview.strip()}")
                    self.emit("tool_result", {"id": tc.id, "name": tc.name, "output": result_str, "is_error": is_error})

                    tool_results.append(
                        ToolResult(
                            call_id=tc.id,
                            name=tc.name,
                            output=result_str,
                            is_error=is_error,
                        )
                    )

                # 3. Add tool execution results to history
                self.history.append(
                    Message(
                        role=Role.TOOL,
                        content="",
                        tool_results=tool_results,
                    )
                )

            else:
                # LLM finished with final answer
                final_text = response.text or "(No response content)"
                self.history.append(
                    Message(role=Role.ASSISTANT, content=final_text)
                )
                self.log(f"\n[ASSISTANT FINAL RESPONSE]:\n{final_text}\n" + "=" * 50)
                self.emit("final_response", {"content": final_text})
                return final_text

        warning = f"Warning: Agent reached maximum steps ({self.max_steps}) without completion."
        self.save_history("history.json")
        self.log(f"{warning} History saved to history.json")
        self.emit("warning", {"message": warning})
        return warning
