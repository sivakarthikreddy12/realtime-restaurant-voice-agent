
import pytest

import order_tools


@pytest.fixture(autouse=True)
def clean_cart():
    """Start every test with an empty cart."""
    order_tools.reset_state()
    yield
    order_tools.reset_state()


@pytest.mark.asyncio
async def test_batch_adds_all_valid_items():
    result = await order_tools._execute_tool_impl(
        "add_to_cart",
        {
            "items": [
                {"item_name": "Cheeseburger", "quantity": 2},
                {"item_name": "Fries", "quantity": 1},
            ]
        },
    )
@pytest.mark.asyncio
async def test_batch_adds_valid_items_and_reports_invalid_items():
    result = await order_tools._execute_tool_impl(
        "add_to_cart",
        {
            "items": [
                {"item_name": "Cheeseburger", "quantity": 2},
                {"item_name": "Steak Burger", "quantity": 1},
                {"item_name": "Coke", "quantity": 1},
            ]
        },
    )

    assert result["status"] == "partial_success"
    assert result["added"] == [
        {"name": "Cheeseburger", "quantity": 2},
        {"name": "Coke", "quantity": 1},
    ]
    assert len(result["invalid_items"]) == 1
    assert result["invalid_items"][0]["item_name"] == "Steak Burger"

    assert order_tools.CART == {
        "Cheeseburger": 2,
        "Coke": 1,
    }
    assert result["subtotal"] == pytest.approx(19.97)

@pytest.mark.asyncio
async def test_batch_with_all_invalid_items_does_not_change_cart():
    result = await order_tools._execute_tool_impl(
        "add_to_cart",
        {
            "items": [
                {"item_name": "Steak Burger", "quantity": 1},
                {"item_name": "Fish Tacos", "quantity": 2},
            ]
        },
    )

    assert result["status"] == "no_items_added"
    assert result["added"] == []
    assert len(result["invalid_items"]) == 2
    assert order_tools.CART == {}
    assert result["subtotal"] == 0

@pytest.mark.asyncio
async def test_invalid_quantity_does_not_block_valid_items():
    result = await order_tools._execute_tool_impl(
        "add_to_cart",
        {
            "items": [
                {"item_name": "Cheeseburger", "quantity": 2},
                {"item_name": "Coke", "quantity": 0},
            ]
        },
    )

    assert result["status"] == "partial_success"
    assert result["added"] == [
        {"name": "Cheeseburger", "quantity": 2}
    ]
    assert result["invalid_items"][0]["item_name"] == "Coke"
    assert "quantity" in result["invalid_items"][0]["reason"].lower()
    assert order_tools.CART == {"Cheeseburger": 2}

@pytest.mark.asyncio
async def test_invalid_batch_entry_does_not_block_valid_items():
    result = await order_tools._execute_tool_impl(
        "add_to_cart",
        {
            "items": [
                {"item_name": "Cheeseburger", "quantity": 1},
                "malformed entry",
            ]
        },
    )

    assert result["status"] == "partial_success"
    assert result["added"] == [
        {"name": "Cheeseburger", "quantity": 1}
    ]
    assert len(result["invalid_items"]) == 1
    assert order_tools.CART == {"Cheeseburger": 1}

@pytest.mark.asyncio
async def test_single_item_call_keeps_legacy_response():
    result = await order_tools._execute_tool_impl(
        "add_to_cart",
        {"item_name": "Coke", "quantity": 2},
    )

    assert result["added"] == {"name": "Coke", "quantity": 2}
    assert "status" not in result
    assert "invalid_items" not in result
    assert order_tools.CART == {"Coke": 2}

@pytest.mark.asyncio
async def test_empty_batch_returns_request_error():
    result = await order_tools._execute_tool_impl(
        "add_to_cart",
        {"items": []},
    )

    assert "error" in result
    assert order_tools.CART == {}
    



# ---------- Boundary and schema tests ----------

@pytest.mark.asyncio
@pytest.mark.parametrize("quantity", [1, 20])
async def test_quantity_boundaries_are_valid(quantity):
    result = await order_tools._execute_tool_impl(
        "add_to_cart",
        {
            "items": [
                {"item_name": "Coke", "quantity": quantity}
            ]
        },
    )

    assert result["status"] == "success"
    assert order_tools.CART["Coke"] == quantity


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "quantity",
    [0, -1, 21, 1.5, True, False],
)
async def test_invalid_quantities_are_rejected(quantity):
    result = await order_tools._execute_tool_impl(
        "add_to_cart",
        {
            "items": [
                {"item_name": "Coke", "quantity": quantity}
            ]
        },
    )

    assert result["status"] == "no_items_added"
    assert result["added"] == []
    assert len(result["invalid_items"]) == 1
    assert order_tools.CART == {}


@pytest.mark.asyncio
async def test_batch_rejects_more_than_20_entries():
    result = await order_tools._execute_tool_impl(
        "add_to_cart",
        {
            "items": [
                {"item_name": "Coke", "quantity": 1}
                for _ in range(21)
            ]
        },
    )

    assert "error" in result
    assert order_tools.CART == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("items", [None, "Coke", {}, 123])
async def test_batch_rejects_non_list_items(items):
    result = await order_tools._execute_tool_impl(
        "add_to_cart",
        {"items": items},
    )

    assert "error" in result
    assert order_tools.CART == {}


@pytest.mark.asyncio
async def test_batch_rejects_missing_item_name():
    result = await order_tools._execute_tool_impl(
        "add_to_cart",
        {
            "items": [
                {"quantity": 1}
            ]
        },
    )

    assert result["status"] == "no_items_added"
    assert result["added"] == []
    assert len(result["invalid_items"]) == 1
    assert order_tools.CART == {}

@pytest.mark.asyncio
async def test_batch_missing_quantity_defaults_to_one():
    result = await order_tools._execute_tool_impl(
        "add_to_cart",
        {
            "items": [
                {"item_name": "Coke"}
            ]
        },
    )

    assert result["status"] == "success"
    assert result["added"] == [
        {"name": "Coke", "quantity": 1}
    ]
    assert order_tools.CART == {"Coke": 1}
    assert result["subtotal"] == pytest.approx(1.99)

def test_add_to_cart_schema_supports_single_and_batch_inputs():
    add_schema = next(
        tool for tool in order_tools.TOOLS
        if tool["name"] == "add_to_cart"
    )

    parameters = add_schema["parameters"]
    properties = parameters["properties"]

    assert "item_name" in properties
    assert "quantity" in properties
    assert "items" in properties

    assert properties["items"]["type"] == "array"
    assert properties["items"]["minItems"] == 1
    assert properties["items"]["maxItems"] == 20

    item_schema = properties["items"]["items"]
    assert item_schema["type"] == "object"
    assert set(item_schema["required"]) == {"item_name"}
    assert "quantity" in item_schema["properties"]
    assert item_schema["additionalProperties"] is False
    assert parameters["additionalProperties"] is False






# ---------- Cart behavior tests ----------

@pytest.mark.asyncio
async def test_adding_same_item_twice_accumulates_quantity():
    await order_tools._execute_tool_impl(
        "add_to_cart",
        {"item_name": "Coke", "quantity": 2},
    )

    result = await order_tools._execute_tool_impl(
        "add_to_cart",
        {
            "items": [
                {"item_name": "Coke", "quantity": 3}
            ]
        },
    )

    assert order_tools.CART["Coke"] == 5
    assert result["subtotal"] == pytest.approx(9.95)


@pytest.mark.asyncio
async def test_duplicate_items_in_batch_accumulate_in_cart():
    result = await order_tools._execute_tool_impl(
        "add_to_cart",
        {
            "items": [
                {"item_name": "Coke", "quantity": 2},
                {"item_name": "Coke", "quantity": 3},
            ]
        },
    )

    assert result["status"] == "success"
    assert order_tools.CART["Coke"] == 5
    assert result["subtotal"] == pytest.approx(9.95)


@pytest.mark.asyncio
async def test_adding_items_preserves_other_cart_items():
    await order_tools._execute_tool_impl(
        "add_to_cart",
        {"item_name": "Coke", "quantity": 2},
    )

    result = await order_tools._execute_tool_impl(
        "add_to_cart",
        {
            "items": [
                {"item_name": "Cheeseburger", "quantity": 1}
            ]
        },
    )

    assert order_tools.CART == {
        "Coke": 2,
        "Cheeseburger": 1,
    }
    assert result["subtotal"] == pytest.approx(12.97)


@pytest.mark.asyncio
async def test_invalid_batch_entry_does_not_change_existing_quantity():
    await order_tools._execute_tool_impl(
        "add_to_cart",
        {"item_name": "Coke", "quantity": 2},
    )

    result = await order_tools._execute_tool_impl(
        "add_to_cart",
        {
            "items": [
                {"item_name": "Coke", "quantity": 3},
                {"item_name": "Unavailable Item", "quantity": 1},
            ]
        },
    )

    assert result["status"] == "partial_success"
    assert order_tools.CART["Coke"] == 5
    assert len(result["invalid_items"]) == 1
    assert result["subtotal"] == pytest.approx(9.95)









# ---------- Confirmation and order-safety tests ----------

@pytest.mark.asyncio
async def test_cannot_create_order_with_empty_cart():
    result = await order_tools._execute_tool_impl(
        "create_mock_order",
        {"customer_confirmation": "yes"},
    )

    assert "error" in result
    assert order_tools.CART == {}


@pytest.mark.asyncio
async def test_cannot_create_order_before_cart_review():
    await order_tools._execute_tool_impl(
        "add_to_cart",
        {"item_name": "Coke", "quantity": 1},
    )

    result = await order_tools._execute_tool_impl(
        "create_mock_order",
        {"customer_confirmation": "yes"},
    )

    assert "error" in result
    assert order_tools.CART == {"Coke": 1}


@pytest.mark.asyncio
async def test_cart_change_invalidates_previous_confirmation():
    await order_tools._execute_tool_impl(
        "add_to_cart",
        {"item_name": "Coke", "quantity": 1},
    )

    # Reviewing the cart establishes the confirmation state.
    await order_tools._execute_tool_impl("get_cart", {})

    # A subsequent change must invalidate that state.
    await order_tools._execute_tool_impl(
        "add_to_cart",
        {"item_name": "French Fries", "quantity": 1},
    )

    result = await order_tools._execute_tool_impl(
        "create_mock_order",
        {"customer_confirmation": "yes"},
    )

    assert "error" in result
    assert order_tools.CART == {
        "Coke": 1,
        "French Fries": 1,
    }


@pytest.mark.asyncio
async def test_cart_review_then_explicit_confirmation_allows_mock_order():
    await order_tools._execute_tool_impl(
        "add_to_cart",
        {"item_name": "Coke", "quantity": 1},
    )

    await order_tools._execute_tool_impl("get_cart", {})

    result = await order_tools._execute_tool_impl(
        "create_mock_order",
        {"customer_confirmation": "yes"},
    )

    assert "error" not in result
    assert result.get("real_order_placed") is False
    assert order_tools.CART == {}
