
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class ToolGraphState(TypedDict):
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


TOOL_NAMES = (
    "show_menu",
    "add_to_cart",
    "remove_from_cart",
    "remove_item_from_cart",
    "set_item_quantity",
    "replace_cart_item",
    "get_cart",
    "create_mock_order",
)


async def route_tool(state: ToolGraphState) -> dict[str, Any]:
    """Validate the requested tool before routing."""
    if state["name"] not in TOOL_NAMES:
        return {
            "result": {"error": f"Unknown tool: {state['name']}"}
        }

    return {}


def choose_tool(state: ToolGraphState) -> str:
    """Choose the graph node matching the requested tool."""
    if state["name"] in TOOL_NAMES:
        return state["name"]

    return "unknown_tool"


async def execute_named_tool(
    state: ToolGraphState,
    tool_name: str,
) -> dict[str, Any]:
    """Delegate to the existing implementation in order_tools.py."""
    from order_tools import _execute_tool_impl

    result = await _execute_tool_impl(
        tool_name,
        state["arguments"],
    )
    return {"result": result}


def make_tool_node(tool_name: str):
    async def tool_node(state: ToolGraphState) -> dict[str, Any]:
        return await execute_named_tool(state, tool_name)

    return tool_node


async def unknown_tool(state: ToolGraphState) -> dict[str, Any]:
    return {
        "result": {"error": f"Unknown tool: {state['name']}"}
    }


builder = StateGraph(ToolGraphState)

builder.add_node("validate_tool", route_tool)

for tool_name in TOOL_NAMES:
    builder.add_node(tool_name, make_tool_node(tool_name))

builder.add_node("unknown_tool", unknown_tool)

builder.add_edge(START, "validate_tool")

builder.add_conditional_edges(
    "validate_tool",
    choose_tool,
    {
        **{name: name for name in TOOL_NAMES},
        "unknown_tool": "unknown_tool",
    },
)

for tool_name in TOOL_NAMES:
    builder.add_edge(tool_name, END)

builder.add_edge("unknown_tool", END)

tool_graph = builder.compile()


async def run_tool_graph(
    name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a tool through LangGraph and return its result."""
    result = await tool_graph.ainvoke(
        {
            "name": name,
            "arguments": arguments or {},
            "result": {},
        }
    )
    return result["result"]
