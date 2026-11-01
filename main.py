import argparse
import sys
import os

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from adapters import get_llm_adapter
from agent import HarnessAgent
from config import DEFAULT_MODEL
from tools import BUILTIN_TOOLS


def main():
    parser = argparse.ArgumentParser(description="Multi-LLM Harness Agent Runner")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model name (e.g. 'gemini-2.5-flash', 'claude-3-7-sonnet-20250219'). Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        type=str,
        help="Prompt for the agent. If omitted, starts interactive chat mode.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Multi-LLM Harness Agent (Adapter Pattern)")
    print(f"Active Model: {args.model}")
    print("=" * 60)

    try:
        adapter = get_llm_adapter(args.model)
    except Exception as e:
        print(f"\n[Initialization Error] {e}")
        sys.exit(1)

    agent = HarnessAgent(adapter=adapter, tools=BUILTIN_TOOLS)
    
    history_file = 'history.json'
    if os.path.exists(history_file):
        print(f"Loading history from {history_file}...")
        try:
            agent.load_history(history_file)
        except Exception as e:
            print(f"Failed to load history: {e}")

    if args.prompt:
        agent.run(args.prompt)
        agent.save_history(history_file)
    else:
        print("\nEntering interactive mode. Type 'exit' or 'quit' to end.\n")
        while True:
            try:
                user_input = input("\nYou > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    print("Exiting.")
                    break
                agent.run(user_input)
                agent.save_history(history_file)
            except KeyboardInterrupt:
                print("\nInterrupted by user. Exiting.")
                break


if __name__ == "__main__":
    main()
