
from main import DEFAULT_INSTRUCTIONS


def test_cart_operations_are_documented():
    assert "remove_item_from_cart" in DEFAULT_INSTRUCTIONS
    assert "set_item_quantity" in DEFAULT_INSTRUCTIONS
    assert "replace_cart_item" in DEFAULT_INSTRUCTIONS
    assert "get_cart" in DEFAULT_INSTRUCTIONS
    assert "create_mock_order" in DEFAULT_INSTRUCTIONS
