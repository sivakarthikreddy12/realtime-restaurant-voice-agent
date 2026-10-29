
import pytest

from tool_graph import run_tool_graph


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name, arguments",
    [
        ("remove_item_from_cart", {"item_name": "Coke"}),
        ("set_item_quantity", {
            "item_name": "Coke",
            "quantity": 2,
        }),
        ("replace_cart_item", {
            "old_item_name": "Coke",
            "new_item_name": "French Fries",
        }),
    ],
)
async def test_new_tools_are_routed(tool_name, arguments):
    result = await run_tool_graph(tool_name, arguments)

    # An empty cart is expected here, so the operation should
    # return a business-level error, not an unknown-tool error.
    assert result.get("error") != f"Unknown tool: {tool_name}"
