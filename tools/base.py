import inspect
from typing import Any, Callable, Dict, Optional, get_type_hints
from adapters.base import ToolSpec


PYTHON_TO_JSON_SCHEMA_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool(name: Optional[str] = None, description: Optional[str] = None) -> Callable:
    """
    Decorator to convert a regular Python function into a ToolSpec with auto-extracted JSON schema.
    """
    def decorator(func: Callable) -> ToolSpec:
        tool_name = name or func.__name__
        tool_desc = description or (inspect.getdoc(func) or "").strip()

        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        properties: Dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            param_type = type_hints.get(param_name, str)
            json_type = PYTHON_TO_JSON_SCHEMA_TYPE.get(param_type, "string")

            properties[param_name] = {
                "type": json_type,
                "description": f"Parameter: {param_name}",
            }

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        return ToolSpec(
            name=tool_name,
            description=tool_desc,
            parameters=parameters,
            handler=func,
        )

    return decorator

