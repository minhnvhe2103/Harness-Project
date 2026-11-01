import os
import subprocess
from tools.base import tool


@tool(
    name="read_file",
    description="Reads and returns the contents of a text file from disk."
)
def read_file(path: str) -> str:
    """Reads the contents of a file given its path."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"


@tool(
    name="write_file",
    description="Writes text content to a file at the specified path (creates or overwrites)."
)
def write_file(path: str, content: str) -> str:
    """Writes content to a file."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{path}'."
    except Exception as e:
        return f"Error writing file '{path}': {str(e)}"


@tool(
    name="list_directory",
    description="Lists files and subdirectories in the specified directory."
)
def list_directory(path: str = ".") -> str:
    """Lists files and folders in directory."""
    try:
        items = os.listdir(path)
        return "\n".join(items) if items else "(empty directory)"
    except Exception as e:
        return f"Error listing directory '{path}': {str(e)}"


@tool(
    name="run_command",
    description="Runs a shell command in the operating system and returns stdout/stderr."
)
def run_command(command: str) -> str:
    """Executes a shell command."""
    try:
        res = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = res.stdout
        if res.stderr:
            output += f"\n[STDERR]\n{res.stderr}"
        if not output.strip():
            output = f"(Command finished with exit code {res.returncode})"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"


BUILTIN_TOOLS = [read_file, write_file, list_directory, run_command]

