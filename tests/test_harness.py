"""
Unit test for HarnessAgent ReAct loop using a Mock LLM Adapter.
"""
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from adapters.base import BaseLLMAdapter, LLMResponse, Message, ToolCall
from agent import HarnessAgent
from tools.builtin import BUILTIN_TOOLS


class MockEchoAdapter(BaseLLMAdapter):
    """A deterministic mock adapter for testing the ReAct cycle."""

    def __init__(self, model_name: str = "mock-llm"):
        super().__init__(model_name=model_name)
        self.step = 0

    def generate(self, messages, tools=None, temperature=0.2, max_tokens=None) -> LLMResponse:
        self.step += 1
        if self.step == 1:
            # First turn: call tool write_file
            return LLMResponse(
                text="I will create a hello_test.txt file.",
                tool_calls=[
                    ToolCall(
                        id="call_001",
                        name="write_file",
                        arguments={"path": "hello_test.txt", "content": "Hello from Harness test!"},
                    )
                ],
            )
        elif self.step == 2:
            # Second turn: call tool read_file
            return LLMResponse(
                text="Now I will read the file to verify.",
                tool_calls=[
                    ToolCall(
                        id="call_002",
                        name="read_file",
                        arguments={"path": "hello_test.txt"},
                    )
                ],
            )
        else:
            # Third turn: final answer
            return LLMResponse(text="Task completed! File verified successfully.")


def test_agent_loop():
    adapter = MockEchoAdapter()
    agent = HarnessAgent(adapter=adapter, tools=BUILTIN_TOOLS)

    result = agent.run("Create and verify hello_test.txt")
    assert "Task completed!" in result

    # Check file exists and clean up
    import os
    assert os.path.exists("hello_test.txt")
    with open("hello_test.txt", "r", encoding="utf-8") as f:
        content = f.read()
    assert content == "Hello from Harness test!"
    os.remove("hello_test.txt")
    print("\n[SUCCESS] Mock Harness Agent loop passed all tests!")


if __name__ == "__main__":
    test_agent_loop()
