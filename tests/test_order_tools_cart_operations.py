
import pytest

import order_tools


@pytest.fixture(autouse=True)
def clean_cart():
    order_tools.reset_state()
    yield
    order_tools.reset_state()


@pytest.mark.asyncio
async def test_remove_entire_item():
    await order_tools._execute_tool_impl(
        "add_to_cart",
        {"item_name": "Coke", "quantity": 3},
    )

    result = await order_tools._execute_tool_impl(
        "remove_item_from_cart",
        {"item_name": "Coke"},
    )

    assert result["removed"]["quantity"] == 3
    assert result["cart"] == []
    assert result["subtotal"] == 0


@pytest.mark.asyncio
async def test_set_exact_quantity():
    await order_tools._execute_tool_impl(
        "add_to_cart",
        {"item_name": "Coke", "quantity": 3},
    )

    result = await order_tools._execute_tool_impl(
        "set_item_quantity",
        {"item_name": "Coke", "quantity": 5},
    )

    coke = next(
        item for item in result["cart"]
        if item["name"] == "Coke"
    )

    assert coke["quantity"] == 5
    assert result["subtotal"] == 9.95


@pytest.mark.asyncio
async def test_replace_item():
    await order_tools._execute_tool_impl(
        "add_to_cart",
        {"item_name": "Coke", "quantity": 3},
    )

    result = await order_tools._execute_tool_impl(
        "replace_cart_item",
        {
            "old_item_name": "Coke",
            "new_item_name": "French Fries",
        },
    )

    assert result["cart"] == [
        {
            "name": "French Fries",
            "quantity": 3,
            "unit_price": 3.49,
            "line_total": 10.47,
        }
    ]
    assert result["subtotal"] == 10.47


@pytest.mark.asyncio
async def test_invalid_replacement_preserves_cart():
    await order_tools._execute_tool_impl(
        "add_to_cart",
        {"item_name": "Coke", "quantity": 3},
    )

    before = await order_tools._execute_tool_impl(
        "get_cart", {}
    )

    result = await order_tools._execute_tool_impl(
        "replace_cart_item",
        {
            "old_item_name": "Coke",
            "new_item_name": "Sprite",
        },
    )

    after = await order_tools._execute_tool_impl(
        "get_cart", {}
    )

    assert "error" in result
    assert after["items"] == before["items"]
    assert after["subtotal"] == before["subtotal"]
