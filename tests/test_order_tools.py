"""Unit tests for the menu/cart tool contract and mock order safety."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import order_tools  # noqa: E402


class ToolSchemaTests(unittest.TestCase):
    def test_tool_definitions_are_valid_and_unique(self) -> None:
        from jsonschema.validators import validator_for

        names = [tool["name"] for tool in order_tools.TOOLS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            names,
            ["show_menu", "add_to_cart", "remove_from_cart", "get_cart", "create_mock_order",
             "remove_item_from_cart",
        "set_item_quantity",
        "replace_cart_item",],
        )
        for tool in order_tools.TOOLS:
            self.assertEqual(tool["type"], "function")
            validator_class = validator_for(tool["parameters"])
            validator_class.check_schema(tool["parameters"])

    def test_module_declares_response_creation(self) -> None:
        self.assertIs(order_tools.CREATE_RESPONSE, True)


class MenuAndCartTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        order_tools.reset_state()

    async def test_show_menu(self) -> None:
        result = await order_tools.execute_tool("show_menu", {})
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(len(result["menu"]), 5)
        self.assertIn("Cheeseburger", [item["name"] for item in result["menu"]])

    async def test_add_by_alias_and_plural(self) -> None:
        result = await order_tools.execute_tool(
            "add_to_cart", {"item_name": "cheeseburgers", "quantity": 2}
        )
        self.assertEqual(result["added"], {"name": "Cheeseburger", "quantity": 2})
        self.assertEqual(result["subtotal"], 17.98)

        result = await order_tools.execute_tool("add_to_cart", {"item_name": "cokes", "quantity": 1})
        self.assertEqual(result["added"], {"name": "Coke", "quantity": 1})
        self.assertEqual(result["subtotal"], 19.97)

        result = await order_tools.execute_tool("add_to_cart", {"item_name": "chips", "quantity": 1})
        self.assertEqual(result["added"]["name"], "French Fries")
        self.assertEqual(result["subtotal"], 23.46)

    async def test_remove_item_and_over_removal(self) -> None:
        await order_tools.execute_tool("add_to_cart", {"item_name": "Cheeseburger", "quantity": 2})
        result = await order_tools.execute_tool(
            "remove_from_cart", {"item_name": "cheeseburger", "quantity": 1}
        )
        self.assertEqual(result["removed"], {"name": "Cheeseburger", "quantity": 1})
        self.assertEqual(result["subtotal"], 8.99)

        result = await order_tools.execute_tool(
            "remove_from_cart", {"item_name": "cheeseburger", "quantity": 5}
        )
        self.assertEqual(result["removed"]["quantity"], 1)
        self.assertIn("Only 1", result["note"])
        self.assertEqual(result["cart"], [])

        result = await order_tools.execute_tool(
            "remove_from_cart", {"item_name": "cheeseburger", "quantity": 1}
        )
        self.assertIn("not in the cart", result["error"])

    async def test_invalid_item_and_quantity(self) -> None:
        result = await order_tools.execute_tool("add_to_cart", {"item_name": "Pizza", "quantity": 1})
        self.assertIn("not on the sample menu", result["error"])
        self.assertIn("Cheeseburger", result["available_items"])

        for quantity in (0, 21, -1, "two", 1.5, True):
            with self.subTest(quantity=quantity):
                result = await order_tools.execute_tool(
                    "add_to_cart", {"item_name": "Coke", "quantity": quantity}
                )
                self.assertIn("whole number", result["error"])

    async def test_get_cart_requires_and_enables_confirmation(self) -> None:
        await order_tools.execute_tool("add_to_cart", {"item_name": "Coke", "quantity": 1})
        result = await order_tools.execute_tool("get_cart", {})
        self.assertIs(result["ready_for_confirmation"], True)
        self.assertEqual(result["subtotal"], 1.99)
        self.assertTrue(order_tools.PENDING_CONFIRMATION)

    async def test_create_order_requires_readback(self) -> None:
        await order_tools.execute_tool("add_to_cart", {"item_name": "Coke", "quantity": 1})
        result = await order_tools.execute_tool(
            "create_mock_order", {"customer_confirmation": "yes, place the order"}
        )
        self.assertIn("Call get_cart", result["error"])

        await order_tools.execute_tool("get_cart", {})
        result = await order_tools.execute_tool(
            "create_mock_order", {"customer_confirmation": "yes, place the order"}
        )
        self.assertEqual(result["status"], "MOCK_ORDER_CREATED")
        self.assertFalse(result["real_order_placed"])
        self.assertTrue(result["order_id"].startswith("DEMO-"))
        self.assertIn("Nothing was sent to Toast", result["note"])
        self.assertEqual(order_tools.CART, {})
        self.assertFalse(order_tools.PENDING_CONFIRMATION)

    async def test_create_order_rejects_negation_and_empty_cart(self) -> None:
        await order_tools.execute_tool("add_to_cart", {"item_name": "Coke", "quantity": 1})
        await order_tools.execute_tool("get_cart", {})
        for refusal in ("no", "not yet", "cancel", "do not place the order", "I don't want it"):
            with self.subTest(refusal=refusal):
                result = await order_tools.execute_tool(
                    "create_mock_order", {"customer_confirmation": refusal}
                )
                self.assertIn("Order not created", result["error"])

        order_tools.reset_state()
        await order_tools.execute_tool("get_cart", {})
        result = await order_tools.execute_tool(
            "create_mock_order", {"customer_confirmation": "place order"}
        )
        self.assertIn("Cart is empty", result["error"])

    async def test_unknown_tool(self) -> None:
        result = await order_tools.execute_tool("nope", {})
        self.assertIn("Unknown tool", result["error"])

    async def test_every_result_is_json_serializable(self) -> None:
        order_tools.reset_state()
        results = [
            await order_tools.execute_tool("show_menu", {}),
            await order_tools.execute_tool("add_to_cart", {"item_name": "Fries", "quantity": 1}),
            await order_tools.execute_tool("get_cart", {}),
            await order_tools.execute_tool("unknown", {}),
        ]
        for result in results:
            self.assertIsInstance(json.loads(json.dumps(result)), dict)


class ConfirmationParserTests(unittest.TestCase):
    def test_accepts_clear_confirmations(self) -> None:
        for text in (
            "yes",
            "Yes please",
            "Yes, place the order.",
            "confirm order",
            "please confirm my order",
            "go ahead",
            "that's correct",
            "okay, submit my order",
            "yes I confirm",
        ):
            with self.subTest(text=text):
                self.assertTrue(order_tools._is_explicit_confirmation(text))

    def test_rejects_refusals_and_empty_values(self) -> None:
        for text in (
            "",
            "no",
            "not yet",
            "cancel",
            "wait",
            "do not place the order",
            "I don't want to order",
            "no thanks",
        ):
            with self.subTest(text=text):
                self.assertFalse(order_tools._is_explicit_confirmation(text))


if __name__ == "__main__":
    unittest.main()
