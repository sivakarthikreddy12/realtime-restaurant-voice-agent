"""Menu and cart tools for the local realtime restaurant voice assistant.

The ``speech-to-speech`` local client expects a module that exposes:

* ``TOOLS``: OpenAI Realtime function-tool definitions.
* ``execute_tool(name, arguments)``: an async callable returning JSON-serializable
  data or a ``ToolResult``.

This module intentionally returns plain dictionaries instead of importing the
package's private ``ToolResult`` class. Plain dictionaries are part of the
documented tool contract, work across ``speech-to-speech`` versions, and avoid a
brittle import of ``speech_to_speech.api.openai_realtime.audio_client``.
"""

from __future__ import annotations

import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

# The local Realtime client asks the model for a follow-up response after every
# tool call, so each tool can report its result back to the customer.
CREATE_RESPONSE = True

MENU: list[dict[str, Any]] = [
    {"name": "Cheeseburger", "price": 8.99, "aliases": ["burger", "cheese burger"]},
    {"name": "Chicken Burger", "price": 9.99, "aliases": ["chicken sandwich"]},
    {"name": "French Fries", "price": 3.49, "aliases": ["fries", "chips"]},
    {"name": "Coke", "price": 1.99, "aliases": ["cola", "coca cola"]},
    {"name": "Vanilla Milkshake", "price": 4.49, "aliases": ["vanilla shake", "milkshake"]},
]

CART: dict[str, int] = {}
PENDING_CONFIRMATION = False

_CART_LOCK = threading.RLock()

_AFFIRMATIVE_EXACT = {
    "yes",
    "yes please",
    "yeah",
    "yep",
    "yup",
    "sure",
    "ok",
    "okay",
    "correct",
    "that is correct",
    "thats correct",
    "sounds good",
    "go ahead",
    "do it",
    "confirm",
    "confirm it",
    "place it",
    "submit it",
}
_CONFIRMATION_PHRASES = (
    "place order",
    "place the order",
    "place my order",
    "submit order",
    "submit the order",
    "submit my order",
    "confirm order",
    "confirm the order",
    "confirm my order",
    "finalize order",
    "finalize the order",
    "complete order",
    "complete the order",
    "go ahead with the order",
    "go ahead and place",
    "that is correct",
    "thats correct",
    "sounds good",
)
_NEGATIONS = {
    "no",
    "not",
    "never",
    "dont",
    "do not",
    "doesnt",
    "does not",
    "cancel",
    "stop",
    "wait",
    "wrong",
    "change",
}


def reset_state() -> None:
    """Clear the cart and confirmation state (used by tests and new sessions)."""

    global PENDING_CONFIRMATION
    with _CART_LOCK:
        CART.clear()
        PENDING_CONFIRMATION = False


def _normalize(text: Any) -> str:
    """Lowercase and collapse punctuation/whitespace for robust matching."""

    value = str(text or "").casefold().replace("'", "").replace("\u2019", "")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _singularize(value: str) -> str:
    """Return a simple English singular form for menu-item matching."""

    if value.endswith("ies") and len(value) > 3:
        return value[:-3] + "y"
    if value.endswith(("ches", "shes", "sses", "xes", "zes")) and len(value) > 3:
        return value[:-2]
    if value.endswith("s") and not value.endswith("ss") and len(value) > 1:
        return value[:-1]
    return value


def _keys(text: Any) -> set[str]:
    normalized = _normalize(text)
    if not normalized:
        return set()
    keys = {normalized, _singularize(normalized)}
    # Also singularize a trailing plural word, e.g. "two cheeseburgers".
    words = normalized.split()
    if words:
        words[-1] = _singularize(words[-1])
        keys.add(" ".join(words))
    return keys


def _find_item(name: Any) -> dict[str, Any] | None:
    """Resolve a menu item by exact name, alias, or simple plural form."""

    query_keys = _keys(name)
    if not query_keys:
        return None
    for item in MENU:
        candidate_keys: set[str] = set()
        for candidate in (item["name"], *item["aliases"]):
            candidate_keys.update(_keys(candidate))
        if query_keys & candidate_keys:
            return item
    return None


def _parse_quantity(value: Any, default: int = 1) -> int | None:
    """Return a valid quantity, or ``None`` when the value is not an integer."""

    if value is None:
        return default
    if isinstance(value, bool):
        return None
    try:
        quantity = int(value)
    except (TypeError, ValueError):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    if quantity < 1 or quantity > 20:
        return None
    return quantity


def _cart_details() -> tuple[list[dict[str, Any]], float]:
    details: list[dict[str, Any]] = []
    total = 0.0
    for name, quantity in CART.items():
        item = _find_item(name)
        if item and quantity > 0:
            line_total = round(item["price"] * quantity, 2)
            details.append(
                {
                    "name": item["name"],
                    "quantity": quantity,
                    "unit_price": item["price"],
                    "line_total": line_total,
                }
            )
            total += line_total
    return details, round(total, 2)


def _menu_names() -> list[str]:
    return [item["name"] for item in MENU]


def _is_explicit_confirmation(value: Any) -> bool:
    """Accept clear affirmations without treating refusal as confirmation."""

    normalized = _normalize(value)
    if not normalized:
        return False
    words = set(normalized.split())
    if words & _NEGATIONS:
        return False
    if normalized in _AFFIRMATIVE_EXACT:
        return True
    if any(phrase in normalized for phrase in _CONFIRMATION_PHRASES):
        return True
    # e.g. "yes, please place my order" or "ok, go ahead"
    if words & {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please"}:
        if words & {"order", "cart", "it", "that", "confirm", "place", "submit"}:
            return True
    return False


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "show_menu",
        "description": "Read the restaurant's sample menu and prices.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "add_to_cart",
        "description": "Add one or more valid menu items to the customer's current cart.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {"type": "string", "description": "Menu item name or alias"},
                "quantity": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["item_name", "quantity"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "remove_from_cart",
        "description": "Remove a quantity of an item from the cart.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {"type": "string", "description": "Menu item name or alias"},
                "quantity": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["item_name", "quantity"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_cart",
        "description": "Read the current cart and subtotal. Call this before asking the customer to confirm an order.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "create_mock_order",
        "description": (
            "Create the fake demo receipt. Call this immediately when the cart was read back and the "
            "customer explicitly says to place or confirm the order. This is the only way to create an "
            "order; never tell the customer the order is placed without calling this tool. Never used for "
            "real restaurant systems."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "customer_confirmation": {
                    "type": "string",
                    "description": "The customer's explicit confirmation, e.g. 'place order' or 'yes, confirm it'.",
                }
            },
            "required": ["customer_confirmation"],
            "additionalProperties": False,
        },
    },
]


async def execute_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute one menu/cart tool and return a JSON-serializable result."""

    global PENDING_CONFIRMATION
    args = arguments or {}
    if not isinstance(args, dict):
        return {"error": "Tool arguments must be a JSON object."}

    if name == "show_menu":
        with _CART_LOCK:
            return {"menu": MENU, "currency": "USD", "note": "Sample menu only; no real restaurant is contacted."}

    if name == "add_to_cart":
        item = _find_item(args.get("item_name", ""))
        quantity = _parse_quantity(args.get("quantity"), default=1)
        if not item:
            return {
                "error": "Item is not on the sample menu.",
                "available_items": _menu_names(),
            }
        if quantity is None:
            return {"error": "Quantity must be a whole number between 1 and 20."}
        with _CART_LOCK:
            CART[item["name"]] = CART.get(item["name"], 0) + quantity
            details, total = _cart_details()
            PENDING_CONFIRMATION = False
            return {
                "added": {"name": item["name"], "quantity": quantity},
                "cart": details,
                "subtotal": total,
            }

    if name == "remove_from_cart":
        item = _find_item(args.get("item_name", ""))
        quantity = _parse_quantity(args.get("quantity"), default=1)
        if not item:
            return {
                "error": "Item is not on the sample menu.",
                "available_items": _menu_names(),
            }
        if quantity is None:
            return {"error": "Quantity must be a whole number between 1 and 20."}
        with _CART_LOCK:
            current = CART.get(item["name"], 0)
            if current <= 0:
                details, total = _cart_details()
                return {
                    "error": f"{item['name']} is not in the cart.",
                    "cart": details,
                    "subtotal": total,
                }
            removed_quantity = min(quantity, current)
            remaining = current - removed_quantity
            if remaining:
                CART[item["name"]] = remaining
            else:
                CART.pop(item["name"], None)
            details, total = _cart_details()
            PENDING_CONFIRMATION = False
            result: dict[str, Any] = {
                "removed": {"name": item["name"], "quantity": removed_quantity},
                "cart": details,
                "subtotal": total,
            }
            if quantity > current:
                result["note"] = f"Only {current} {item['name']} were in the cart."
            return result

    if name == "get_cart":
        with _CART_LOCK:
            details, total = _cart_details()
            PENDING_CONFIRMATION = bool(details)
            return {
                "items": details,
                "subtotal": total,
                "ready_for_confirmation": PENDING_CONFIRMATION,
                "next_step": (
                    "Read every item, quantity, and the subtotal back to the customer, then ask them to "
                    "explicitly confirm before calling create_mock_order."
                    if details
                    else "The cart is empty. Ask the customer what they would like to add."
                ),
            }

    if name == "create_mock_order":
        confirmation = args.get("customer_confirmation", "")
        if not _is_explicit_confirmation(confirmation):
            return {
                "error": (
                    "Order not created. The customer must explicitly say something like 'place order' or "
                    "'confirm order' first."
                )
            }
        with _CART_LOCK:
            details, total = _cart_details()
            if not details:
                return {"error": "Cart is empty; no order created."}
            if not PENDING_CONFIRMATION:
                return {
                    "error": (
                        "Order not created yet. Call get_cart, read the full cart and subtotal back to the "
                        "customer, and ask for explicit confirmation first."
                    )
                }
            receipt = {
                "status": "MOCK_ORDER_CREATED",
                "order_id": "DEMO-" + uuid.uuid4().hex[:6].upper(),
                "items": details,
                "subtotal": total,
                "currency": "USD",
                "real_order_placed": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "note": "Sample order only. Nothing was sent to Toast or a real restaurant.",
            }
            CART.clear()
            PENDING_CONFIRMATION = False
        return receipt

    return {"error": f"Unknown tool: {name}"}
