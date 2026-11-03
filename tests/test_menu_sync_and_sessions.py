from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import menu_sync
import order_tools


def sample_html() -> str:
    rows = "".join(
        f'<div data-hook="item.root"><p data-hook="item.name">Dish {i}</p>'
        f'<p data-hook="item.description">Description {i}</p>'
        f'<p data-hook="item.price">${i}.00</p></div>'
        for i in range(1, 12)
    )
    return f"<h2>Test Category</h2>{rows}<div data-hook='item.root'><p data-hook='item.name'>No price item</p></div>"


class MenuSyncTests(unittest.IsolatedAsyncioTestCase):
    def test_parser_reads_rows_and_retains_missing_prices(self) -> None:
        items = menu_sync.parse_menu_html(sample_html())
        self.assertEqual(len(items), 12)
        self.assertEqual(items[0]["category"], "Test Category")
        self.assertEqual(items[0]["price"], 1.0)
        self.assertIsNone(items[-1]["price"])

    def test_parser_rejects_partial_or_changed_markup(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected at least 10"):
            menu_sync.parse_menu_html("<main>new page without menu hooks</main>")

    def test_live_menu_snapshot_contains_prices_options_and_unpriced_rows(self) -> None:
        snapshot = json.loads(Path("tests/fixtures/sanpeggio_menu_snapshot.json").read_text())
        rows = snapshot["items"]
        self.assertEqual(len(rows), 66)
        self.assertEqual(rows[0]["source_url"], menu_sync.SOURCE_URL)
        byo = next(row for row in rows if row["name"] == "Build Your Own Pizza - 14” Large")
        self.assertEqual(byo["price"], 14.49)
        self.assertIn("Pepperoni", byo["customizations"]["toppings"]["choices"])
        self.assertTrue(any(row["price"] is None for row in rows))

    async def test_sync_sends_complete_menu_to_store_upsert(self) -> None:
        import postgres_store
        rows = menu_sync.parse_menu_html(sample_html())
        with (
            patch.object(menu_sync, "fetch_menu", return_value=rows),
            patch.object(postgres_store, "upsert_store_menu", new=AsyncMock()) as upsert,
        ):
            result = await menu_sync.sync_store("test-location", "https://example.test/menu")
        self.assertEqual(result["status"], "success")
        upsert.assert_awaited_once_with("test-location", rows, source_url="https://example.test/menu")

    def test_customization_validator_rejects_unknown_choices_and_prices_valid_ones(self) -> None:
        item = {
            "name": "Build Your Own Pizza - 14 inch Large", "price": 14.49,
            "size": "14 inch Large", "aliases": [],
            "customizations": {
                "crust": {"choices": ["New York Style", "Thin & Crispy"],
                          "price_deltas": {"Thin & Crispy": 2.0},
                          "size_limits": {"Thin & Crispy": [14]}},
                "toppings": {"choices": ["Pepperoni"], "price_by_size": {"large": 2.25}},
            },
        }
        valid, surcharge, error = order_tools._validated_modifiers(
            item, {"crust": "Thin & Crispy", "toppings": ["Pepperoni"]}
        )
        self.assertIsNone(error)
        self.assertEqual(valid, {"crust": "Thin & Crispy", "toppings": ["Pepperoni"]})
        self.assertEqual(surcharge, 4.25)
        _, _, error = order_tools._validated_modifiers(item, {"toppings": ["Coffee"]})
        self.assertIn("not listed", error)


class SessionIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        order_tools.reset_state()

    async def test_two_sessions_have_independent_carts_and_reviews(self) -> None:
        with order_tools.use_session("session-a"):
            await order_tools.execute_tool("add_to_cart", {"item_name": "Coke", "quantity": 2})
            await order_tools.execute_tool("get_cart", {})
        with order_tools.use_session("session-b"):
            await order_tools.execute_tool("add_to_cart", {"item_name": "French Fries", "quantity": 1})
            cart_b = await order_tools.execute_tool("get_cart", {})
        with order_tools.use_session("session-a"):
            cart_a = await order_tools.execute_tool("get_cart", {})
        self.assertEqual(cart_a["items"][0]["name"], "Coke")
        self.assertEqual(cart_a["subtotal"], 3.98)
        self.assertEqual(cart_b["items"][0]["name"], "French Fries")
        self.assertEqual(cart_b["subtotal"], 3.49)

    async def test_session_welcome_is_returned_once(self) -> None:
        session = "welcome-once"
        first = await order_tools.session_welcome(session)
        second = await order_tools.session_welcome(session)
        self.assertIn("Sanpeggio's Pizza", first)
        self.assertIsNone(second)

    async def test_session_expiry_removes_old_idle_session(self) -> None:
        with patch.dict(os.environ, {"SESSION_TTL_SECONDS": "60"}):
            state = order_tools._session_state("expires")
            state.last_activity -= 120
            order_tools._session_state("another")
            self.assertNotIn("expires", order_tools._SESSION_STATES)

    async def test_unpriced_catalog_item_cannot_enter_cart(self) -> None:
        token = order_tools._ACTIVE_MENU.set([
            {"name": "Unpriced Special", "price": None, "aliases": [], "available": True}
        ])
        try:
            result = await order_tools.execute_tool(
                "add_to_cart", {"item_name": "Unpriced Special", "quantity": 1}
            )
        finally:
            order_tools._ACTIVE_MENU.reset(token)
        self.assertIn("no published price", result["error"])
        self.assertEqual(order_tools.CART, {})

    async def test_validated_size_topping_and_price_enter_cart(self) -> None:
        item = {
            "name": "Build Your Own Pizza - 14 inch Large", "price": 14.49,
            "size": "14 inch Large", "aliases": ["large pizza"], "available": True,
            "customizations": {
                "crust": {"choices": ["New York Style", "Thin & Crispy"],
                          "price_deltas": {"Thin & Crispy": 2.0},
                          "size_limits": {"Thin & Crispy": [14]}},
                "toppings": {"choices": ["Pepperoni"], "price_by_size": {"large": 2.25}},
            },
        }
        token = order_tools._ACTIVE_MENU.set([item])
        try:
            result = await order_tools.execute_tool("add_to_cart", {
                "item_name": "large thin and crispy pepperoni pizza", "quantity": 1,
            })
            self.assertEqual(result["subtotal"], 18.74)
            self.assertEqual(result["cart"][0]["customizations"]["toppings"], ["Pepperoni"])
        finally:
            order_tools._ACTIVE_MENU.reset(token)

    async def test_pizza_without_size_requests_a_size(self) -> None:
        rows = json.loads(Path("tests/fixtures/sanpeggio_menu_snapshot.json").read_text())["items"]
        token = order_tools._ACTIVE_MENU.set(rows)
        try:
            result = order_tools._resolve_requested_item("pepperoni pizza", None)
        finally:
            order_tools._ACTIVE_MENU.reset(token)
        self.assertIn("choose a pizza size", result[2])

    async def test_unknown_store_does_not_receive_another_stores_fallback_menu(self) -> None:
        import postgres_store
        with (
            patch.object(postgres_store, "configured", return_value=False),
            patch.object(postgres_store, "is_open", return_value=False),
            order_tools.use_session("unconfigured-store", "store-two"),
        ):
            result = await order_tools.execute_tool("show_menu", {})
            add = await order_tools.execute_tool("add_to_cart", {"item_name": "Coke"})
        self.assertEqual(result["menu"], [])
        self.assertIn("not on this store's menu", add["error"])

    async def test_live_menu_variant_resolves_natural_size_and_topping_request(self) -> None:
        rows = json.loads(Path("tests/fixtures/sanpeggio_menu_snapshot.json").read_text())["items"]
        token = order_tools._ACTIVE_MENU.set(rows)
        try:
            with order_tools.use_session("parsed-live-menu"):
                result = await order_tools.execute_tool("add_to_cart", {
                    "item_name": "large thin and crispy pepperoni pizza", "quantity": 1,
                })
            self.assertEqual(result["subtotal"], 18.74)
            self.assertEqual(result["cart"][0]["customizations"], {
                "crust": "Thin & Crispy", "toppings": ["Pepperoni"]
            })
        finally:
            order_tools._ACTIVE_MENU.reset(token)

    async def test_database_failure_does_not_clear_reviewed_cart(self) -> None:
        import postgres_store

        order_tools._STORE_MENU_CACHE.pop("sanpeggios-demo", None)
        with order_tools.use_session("failed-commit"):
            await order_tools.execute_tool("add_to_cart", {"item_name": "Coke", "quantity": 1})
            await order_tools.execute_tool("get_cart", {})
            with (
                patch.object(postgres_store, "is_open", return_value=True),
                patch.object(postgres_store, "load_store_menu", new=AsyncMock(return_value=order_tools.MENU)),
                patch.object(postgres_store, "create_order", new=AsyncMock(side_effect=RuntimeError("database unavailable"))),
                order_tools.use_user_message("yes, place the order"),
            ):
                result = await order_tools.execute_tool(
                    "create_mock_order", {"customer_confirmation": "yes, place the order"}
                )
            self.assertIn("cart is still active", result["error"])
            self.assertEqual((await order_tools.execute_tool("get_cart", {}))["subtotal"], 1.99)


if __name__ == "__main__":
    unittest.main()
