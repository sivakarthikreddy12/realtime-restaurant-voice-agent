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

import asyncio
import os
import re
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Callable

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

_DEFAULT_SESSION_ID = "00000000-0000-4000-8000-000000000000"
CART: dict[str, int] = {}
PENDING_CONFIRMATION = False


@dataclass
class _OrderSession:
    session_id: str
    store_id: str = "sanpeggios-demo"
    cart: dict[str, int] = field(default_factory=dict)
    pending_confirmation: bool = False
    state: str = "ACTIVE"
    revision: int = 0
    reviewed_revision: int = -1
    welcomed: bool = False
    last_activity: float = field(default_factory=lambda: __import__("time").monotonic())


_DEFAULT_STATE = _OrderSession(_DEFAULT_SESSION_ID, cart=CART)
_SESSION_STATES: dict[str, _OrderSession] = {_DEFAULT_SESSION_ID: _DEFAULT_STATE}

_ACTIVE_SESSION_ID: ContextVar[str] = ContextVar("active_order_session_id", default=_DEFAULT_SESSION_ID)
_ACTIVE_STORE_ID: ContextVar[str] = ContextVar("active_order_store_id", default="sanpeggios-demo")
_ACTIVE_MENU: ContextVar[list[dict[str, Any]]] = ContextVar("active_order_menu", default=MENU)
_CURRENT_USER_MESSAGE: ContextVar[str | None] = ContextVar("current_order_user_message", default=None)
_LATENCY_CALLBACK: ContextVar[Callable[..., None] | None] = ContextVar("order_latency_callback", default=None)
_DATABASE_TOOL_LOCK = asyncio.Lock()
_STORE_MENU_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}

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
_ORDER_EDIT_REQUEST = re.compile(
    r"\b(add|remove|delete|replace|swap|change|update|set|increase|decrease|"
    r"i want|i would like|id like|i d like|can i get|could i get|give me|take off)\b",
    re.IGNORECASE,
)
_EXPLICIT_CART_ACTION = re.compile(
    r"\b(add|remove|delete|replace|swap|change|update|set|increase|decrease|take off)\b",
    re.IGNORECASE,
)
_ITEM_REQUEST = re.compile(
    r"\b(id like|i d like|i want|can i get|could i get|can i have|could i have|give me)\b",
    re.IGNORECASE,
)
_ORDER_REVIEW_QUESTIONS = (
    "can you ", "could you ", "would you ", "will you ", "should i ", "do you ",
    "what ", "how ", "read back ", "tell me ",
)


def reset_state() -> None:
    """Clear the cart and confirmation state (used by tests and new sessions)."""

    global PENDING_CONFIRMATION
    with _CART_LOCK:
        CART.clear()
        PENDING_CONFIRMATION = False
        _DEFAULT_STATE.pending_confirmation = False
        _DEFAULT_STATE.revision = 0
        _DEFAULT_STATE.reviewed_revision = -1


def _session_state(session_id: str | None = None) -> _OrderSession:
    sid = session_id or _ACTIVE_SESSION_ID.get()
    now = __import__("time").monotonic()
    ttl = max(60, int(os.getenv("SESSION_TTL_SECONDS", "7200")))
    for key, value in tuple(_SESSION_STATES.items()):
        if key != _DEFAULT_SESSION_ID and now - value.last_activity > ttl:
            _SESSION_STATES.pop(key, None)
    state = _SESSION_STATES.get(sid)
    if state is None:
        state = _OrderSession(sid, store_id=_ACTIVE_STORE_ID.get())
        _SESSION_STATES[sid] = state
    state.last_activity = now
    return state


def _cart() -> dict[str, int]:
    return _session_state().cart


def _pending() -> bool:
    return _session_state().pending_confirmation


def _set_pending(value: bool) -> None:
    global PENDING_CONFIRMATION
    state = _session_state()
    state.pending_confirmation = bool(value)
    state.state = "REVIEWED" if value else "ACTIVE"
    if not value:
        state.revision += 1
    state.reviewed_revision = state.revision if value else -1
    if state.session_id == _DEFAULT_SESSION_ID:
        PENDING_CONFIRMATION = bool(value)


@contextmanager
def use_session(session_id: str, store_id: str = "sanpeggios-demo"):
    """Bind tool calls to an isolated in-memory customer session and store."""
    token = _ACTIVE_SESSION_ID.set(session_id)
    store_token = _ACTIVE_STORE_ID.set(store_id)
    _session_state(session_id).store_id = store_id
    try:
        yield
    finally:
        _ACTIVE_STORE_ID.reset(store_token)
        _ACTIVE_SESSION_ID.reset(token)


@contextmanager
def use_user_message(message: str | None):
    """Bind tool calls to the latest customer utterance for checkout validation."""

    token = _CURRENT_USER_MESSAGE.set(message)
    try:
        yield
    finally:
        _CURRENT_USER_MESSAGE.reset(token)


@contextmanager
def use_latency_callback(callback: Callable[..., None] | None):
    token = _LATENCY_CALLBACK.set(callback)
    try:
        yield
    finally:
        _LATENCY_CALLBACK.reset(token)


def _log_database_latency(operation: str, started: float) -> None:
    callback = _LATENCY_CALLBACK.get()
    if callback:
        callback("database_complete", operation=operation,
                 duration=(__import__("time").perf_counter() - started) * 1000)


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
    for item in _ACTIVE_MENU.get():
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


def _validated_modifiers(item: dict[str, Any], requested: Any) -> tuple[dict[str, Any], float, str | None]:
    if requested in (None, {}):
        return {}, 0.0, None
    if not isinstance(requested, dict):
        return {}, 0.0, "Customizations must be an object."
    options = item.get("customizations") or {}
    if not options:
        return {}, 0.0, "This menu item has no published customization options."
    unknown = set(requested) - {"crust", "toppings"}
    if unknown:
        return {}, 0.0, f"Unsupported customization: {sorted(unknown)[0]}."
    result: dict[str, Any] = {}
    surcharge = 0.0
    if "crust" in requested:
        crusts = options.get("crust", {})
        crust = next((x for x in crusts.get("choices", []) if _normalize(x) == _normalize(requested["crust"])), None)
        if not crust:
            return {}, 0.0, f"Crust '{requested['crust']}' is not available for this pizza."
        size = re.search(r"(\d+)", str(item.get("size") or item["name"]))
        inches = int(size.group(1)) if size else None
        limits = crusts.get("size_limits", {}).get(crust)
        if limits and inches not in limits:
            return {}, 0.0, f"{crust} is not available for this pizza size."
        result["crust"] = crust
        surcharge += float(crusts.get("price_deltas", {}).get(crust, 0))
    toppings = requested.get("toppings", [])
    if not isinstance(toppings, list) or len(toppings) > 20:
        return {}, 0.0, "Toppings must be a list of up to 20 choices."
    if toppings:
        choices = options.get("toppings", {})
        size = re.search(r"(\d+)", str(item.get("size") or item["name"]))
        size_name = {7: "small", 12: "medium", 14: "large", 16: "x-large"}.get(int(size.group(1))) if size else None
        unit = choices.get("price_by_size", {}).get(size_name)
        if unit is None:
            return {}, 0.0, "A topping price is not published for this pizza size."
        selected = []
        for raw in toppings:
            choice = next((x for x in choices.get("choices", []) if _normalize(x) == _normalize(raw)), None)
            if not choice:
                return {}, 0.0, f"Topping '{raw}' is not listed for this pizza."
            selected.append(choice)
            surcharge += float(unit)
        result["toppings"] = selected
    return result, round(surcharge, 2), None


def _line_key(name: str, modifiers: dict[str, Any]) -> str:
    if not modifiers:
        return name
    import json
    return name + "||" + json.dumps(modifiers, sort_keys=True, separators=(",", ":"))


def _resolve_requested_item(name: Any, supplied: Any) -> tuple[dict[str, Any] | None, dict[str, Any], str | None]:
    """Resolve exact menu variants and infer only explicitly listed options."""
    direct = _find_item(name)
    if direct:
        return direct, supplied or {}, None
    if supplied is not None and not isinstance(supplied, dict):
        return None, {}, None
    words = _normalize(name).split()
    fillers = {"with", "and", "a", "an", "the"}
    choices = []
    for candidate in _ACTIVE_MENU.get():
        options = candidate.get("customizations") or {}
        if not options:
            continue
        topping_hits: list[str] = []
        crust_hit: str | None = None
        remaining = [word for word in words if word not in fillers]
        for topping in sorted(options.get("toppings", {}).get("choices", []), key=len, reverse=True):
            tokens = [x for x in _normalize(topping).split() if x not in fillers]
            for offset in range(len(remaining) - len(tokens) + 1):
                if remaining[offset:offset + len(tokens)] == tokens:
                    del remaining[offset:offset + len(tokens)]
                    topping_hits.append(topping)
                    break
        for crust in options.get("crust", {}).get("choices", []):
            tokens = [x for x in _normalize(crust).split() if x not in fillers]
            for offset in range(len(remaining) - len(tokens) + 1):
                if remaining[offset:offset + len(tokens)] == tokens:
                    del remaining[offset:offset + len(tokens)]
                    crust_hit = crust
                    break
        remaining = [word for word in remaining if word not in fillers]
        base = " ".join(remaining)
        if base and _find_item(base) is candidate:
            inferred: dict[str, Any] = {}
            if topping_hits:
                inferred["toppings"] = topping_hits
            if crust_hit:
                inferred["crust"] = crust_hit
            choices.append((candidate, inferred))
    if len(choices) == 1:
        return choices[0][0], choices[0][1], None
    if not choices and "pizza" in words and not any(w in words for w in ("small", "medium", "large", "xl", "extra")):
        sizes = [item["name"] for item in _ACTIVE_MENU.get()
                 if item.get("customizations", {}).get("toppings")]
        if sizes:
            return None, {}, "Please choose a pizza size. Available options: " + ", ".join(sizes)
    return None, {}, None


def _keys_for_item(name: str) -> list[str]:
    return [key for key in _cart() if (key.partition("||")[0] == name)]


def _quantity_for_item(name: str) -> int:
    return sum(_cart().get(key, 0) for key in _keys_for_item(name))


def _remove_quantity_for_item(name: str, quantity: int) -> int:
    remaining = quantity
    removed = 0
    for key in _keys_for_item(name):
        current = _cart().get(key, 0)
        amount = min(current, remaining)
        if amount:
            if current == amount:
                _cart().pop(key, None)
            else:
                _cart()[key] = current - amount
            remaining -= amount
            removed += amount
        if not remaining:
            break
    return removed


def _cart_details() -> tuple[list[dict[str, Any]], float]:
    details: list[dict[str, Any]] = []
    total = 0.0
    for line_key, quantity in _cart().items():
        name, separator, encoded = line_key.partition("||")
        modifiers: dict[str, Any] = {}
        if separator:
            import json
            try:
                modifiers = json.loads(encoded)
            except ValueError:
                continue
        item = _find_item(name)
        if item and quantity > 0:
            if item.get("price") is None:
                continue
            checked_modifiers, surcharge, modifier_error = _validated_modifiers(item, modifiers)
            if modifier_error:
                continue
            unit_price = round(float(item["price"]) + surcharge, 2)
            line_total = round(unit_price * quantity, 2)
            details.append({
                    "name": item["name"],
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "line_total": line_total,
                    **({"customizations": checked_modifiers} if checked_modifiers else {}),
            })
            total += line_total
    return details, round(total, 2)


def _menu_names() -> list[str]:
    return [item["name"] for item in _ACTIVE_MENU.get()]


def _is_explicit_confirmation(value: Any) -> bool:
    """Accept clear affirmations without treating refusal as confirmation."""

    normalized = _normalize(value)
    if not normalized:
        return False
    if any(normalized.startswith(prefix) for prefix in _ORDER_REVIEW_QUESTIONS):
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


def message_requires_tool(value: Any) -> bool:
    """Identify user requests that should be answered from the menu/cart tools."""

    normalized = _normalize(value)
    if _EXPLICIT_CART_ACTION.search(normalized):
        return True
    if re.search(r"\b(menu|cart|subtotal|place the order|place my order|submit order)\b", normalized):
        return True
    if re.search(
        r"\b(confirm|review|read back|show me|what(?:s| is| are)?).{0,30}\b(order|cart|items|this|that|it|one)\b",
        normalized,
    ):
        return True
    if _ITEM_REQUEST.search(normalized):
        # A generic wish to start ordering needs a conversational prompt; an
        # item request should be checked against the actual menu.
        if re.search(r"\b(place|start|make) an? order\b", normalized):
            return False
        return True
    return False


async def invalidate_review_for_message(value: Any) -> None:
    """Invalidate a reviewed cart when the customer asks to change its contents."""

    global PENDING_CONFIRMATION
    if not _ORDER_EDIT_REQUEST.search(_normalize(value)):
        return

    with _CART_LOCK:
        _set_pending(False)


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "show_menu",
        "description": "Read the active store's menu and authoritative prices.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },

{
    "type": "function",
    "name": "add_to_cart",
    "description": (
        "Add one or more valid menu items to the customer's current cart. "
        "Use item_name and quantity for a single item, or items for "
        "multiple items in one call. If quantity is omitted, default to 1."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "item_name": {
                "type": "string",
                "description": "Single menu item name or alias."
            },
            "quantity": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "description": "Quantity for the single item; defaults to 1."
            },
            "customizations": {
                "type": "object",
                "properties": {
                    "crust": {"type": "string"},
                    "toppings": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": 20,
                "description": "Multiple items to add in one call.",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_name": {
                            "type": "string",
                            "description": "Menu item name or alias."
                        },
                        "quantity": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "description": "Quantity; defaults to 1."
                        },
                        "customizations": {
                            "type": "object",
                            "properties": {
                                "crust": {"type": "string"},
                                "toppings": {"type": "array", "items": {"type": "string"}},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "required": ["item_name"],
                    "additionalProperties": False
                }
            }
        },
        "required": [],
        "additionalProperties": False
    }
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
    
    
    
    
    {
        "type": "function",
        "name": "remove_item_from_cart",
        "description": (
            "Remove an entire menu item from the cart, regardless of its "
            "current quantity. Use when the customer says remove all, "
            "remove the item completely, or delete the item."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "Menu item to remove completely."
                }
            },
            "required": ["item_name"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "set_item_quantity",
        "description": (
            "Set an item's final quantity to the requested number. "
            "Do not add to its existing quantity. Use when the customer "
            "says make it, change it to, or set it to a specific quantity."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "Menu item to update."
                },
                "quantity": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Desired final quantity."
                }
            },
            "required": ["item_name", "quantity"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "replace_cart_item",
        "description": (
            "Replace an item already in the cart with another valid menu "
            "item. Preserve the old item's quantity. If the replacement "
            "item is already in the cart, combine the quantities."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "old_item_name": {
                    "type": "string",
                    "description": "Existing item to replace."
                },
                "new_item_name": {
                    "type": "string",
                    "description": "Replacement item from the menu."
                }
            },
            "required": ["old_item_name", "new_item_name"],
            "additionalProperties": False,
        },
    },

]


async def _execute_tool_memory(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute one menu/cart tool and return a JSON-serializable result."""

    global PENDING_CONFIRMATION
    args = arguments or {}
    if not isinstance(args, dict):
        return {"error": "Tool arguments must be a JSON object."}

    if name == "show_menu":
        import postgres_store
        if postgres_store.is_open():
            state = _session_state()
            catalog = await postgres_store.load_store_menu_catalog(state.store_id)
            if not catalog.get("categories"):
                return {
                    "error": (
                        "The database has no synced menu for this store. Do not name or guess menu "
                        "items. Ask the customer to try again after the menu has synced."
                    ),
                    "store_id": state.store_id,
                }
            customer_categories = []
            for category in catalog.get("categories", []):
                customer_items = []
                for product in category.get("items", []):
                    variants = [
                        {
                            "name": variant.get("size") or variant.get("name"),
                            "price": variant.get("price"),
                            "price_status": variant.get("price_status", "not_listed"),
                        }
                        for variant in product.get("variants", [])
                    ]
                    options = [
                        {
                            "name": option["name"],
                            "choices": option.get("choices", []),
                            "constraints": option.get("constraints", {}),
                        }
                        for option in product.get("options", [])
                    ]
                    customer_items.append({
                        "name": product["name"],
                        "description": product.get("description", ""),
                        "availability": "available" if product.get("available", True) else "unavailable",
                        "prices": variants,
                        "options": options,
                    })
                customer_categories.append({
                    "name": category["name"],
                    "notes": category.get("notes", []),
                    "items": customer_items,
                })
            return {
                "menu": {
                    "restaurant": catalog.get("store", {}).get("name"),
                    "currency": catalog.get("currency", "USD"),
                    "categories": customer_categories,
                },
                "note": (
                    "Use only the exact category and item names in this menu. Do not add categories, "
                    "items, prices, or examples from memory. A null price or price_status 'not_listed' "
                    "means the published menu does not give a price; say so instead of guessing."
                ),
            }
        with _CART_LOCK:
            return {"menu": _ACTIVE_MENU.get(), "currency": "USD", "note": "Menu data is for this demo; no real restaurant is contacted."}

    
    
    if name == "add_to_cart":
    # Support both batch requests and the existing single-item format.
        is_batch = "items" in args

        if is_batch:
            if "item_name" in args or "quantity" in args:
                return {
                "error": (
                    "Provide either 'items' or the single-item "
                    "arguments 'item_name' and 'quantity', not both."
                    )
                }

            requested_items = args["items"]

            if not isinstance(requested_items, list) or not requested_items:
                return {"error": "'items' must be a non-empty list."}

            if len(requested_items) > 20:
                return {
                    "error": (
                        "A maximum of 20 item entries can be added "
                        "in one call."
                    )
                }
        else:
        # Keep the legacy single-item input format working.
            requested_items = [{
                "item_name": args.get("item_name", ""),
                "quantity": args.get("quantity"),
                "customizations": args.get("customizations"),
            }]

        validated_items: list[tuple[dict[str, Any], int, dict[str, Any], float]] = []
        invalid_items: list[dict[str, str]] = []

        # Validate every entry. In batch mode, one invalid entry
        # must not prevent other valid entries from being added.
        for requested in requested_items:
            if not isinstance(requested, dict):
                if is_batch:
                    invalid_items.append({
                        "item_name": str(requested),
                        "reason": (
                            "Each entry must be an object containing "
                            "'item_name' and 'quantity'."
                        ),
                    })
                    continue

                return {
                    "error": "The item must be a valid JSON object."
                }

            item_name = requested.get("item_name", "")
            item, inferred_modifiers, resolution_error = _resolve_requested_item(
                item_name, requested.get("customizations")
            )
            requested_modifiers = dict(inferred_modifiers)
            if isinstance(requested.get("customizations"), dict):
                for key, value in requested["customizations"].items():
                    if key == "toppings" and key in requested_modifiers and isinstance(value, list):
                        requested_modifiers[key] = list(dict.fromkeys(requested_modifiers[key] + value))
                    else:
                        requested_modifiers[key] = value
            quantity = _parse_quantity(requested.get("quantity"))

            if not item:
                reason = resolution_error or "Item is not on this store's menu."

                if is_batch:
                    invalid_items.append({
                        "item_name": str(item_name),
                        "reason": reason,
                    })
                    continue

                return {
                    "error": reason,
                    "available_items": _menu_names(),
                }

            if item.get("price") is None:
                reason = "This menu item has no published price available for ordering."
                if is_batch:
                    invalid_items.append({"item_name": str(item_name), "reason": reason})
                    continue
                return {"error": reason}
            if not item.get("available", True):
                reason = "This menu item is currently unavailable."
                if is_batch:
                    invalid_items.append({"item_name": str(item_name), "reason": reason})
                    continue
                return {"error": reason}

            if quantity is None:
                reason = (
                    "Quantity must be a whole number between 1 and 20."
                )

                if is_batch:
                    invalid_items.append({
                        "item_name": item["name"],
                        "reason": reason,
                    })
                    continue

                return {"error": reason}

            modifiers, surcharge, modifier_error = _validated_modifiers(
                item, requested_modifiers
            )
            if modifier_error:
                if is_batch:
                    invalid_items.append({"item_name": str(item_name), "reason": modifier_error})
                    continue
                return {"error": modifier_error}

            validated_items.append((item, quantity, modifiers, surcharge))

        # Add all valid entries together while holding the cart lock.
        # If no entries are valid, the cart remains unchanged.
        with _CART_LOCK:
            if validated_items:
                for item, quantity, modifiers, _surcharge in validated_items:
                    canonical_name = _line_key(item["name"], modifiers)
                    _cart()[canonical_name] = (
                        _cart().get(canonical_name, 0) + quantity
                    )

            # The cart changed, so require confirmation again.
                _set_pending(False)

            details, total = _cart_details()

        added_items = [
            {"name": item["name"], "quantity": quantity,
             **({"customizations": modifiers} if modifiers else {})}
            for item, quantity, modifiers, _surcharge in validated_items
        ]

        if not is_batch:
        # Preserve the legacy single-item response shape.
            return {
                "added": added_items[0],
                "cart": details,
                "subtotal": total,
            }

        if added_items and invalid_items:
            status = "partial_success"
        elif added_items:
            status = "success"
        else:
            status = "no_items_added"

        return {
            "status": status,
            "added": added_items,
            "invalid_items": invalid_items,
            "cart": details,
            "subtotal": total,
        }
    
    if name == "remove_from_cart":
        item = _find_item(args.get("item_name", ""))
        quantity = _parse_quantity(args.get("quantity"), default=1)
        if not item:
            return {
                "error": "Item is not on this store's menu.",
                "available_items": _menu_names(),
            }
        if quantity is None:
            return {"error": "Quantity must be a whole number between 1 and 20."}
        with _CART_LOCK:
            current = _quantity_for_item(item["name"])
            if current <= 0:
                details, total = _cart_details()
                return {
                    "error": f"{item['name']} is not in the cart.",
                    "cart": details,
                    "subtotal": total,
                }
            removed_quantity = _remove_quantity_for_item(item["name"], min(quantity, current))
            details, total = _cart_details()
            _set_pending(False)
            result: dict[str, Any] = {
                "removed": {"name": item["name"], "quantity": removed_quantity},
                "cart": details,
                "subtotal": total,
            }
            if quantity > current:
                result["note"] = f"Only {current} {item['name']} were in the cart."
            return result

    
    if name == "remove_item_from_cart":
        item = _find_item(args.get("item_name", ""))

        if not item:
            return {
                "error": "Item is not on this store's menu.",
                "available_items": _menu_names(),
            }

        with _CART_LOCK:
            current = _quantity_for_item(item["name"])

            if current == 0:
                details, total = _cart_details()
                return {
                    "error": f"{item['name']} is not in the cart.",
                    "cart": details,
                    "subtotal": total,
                }

            for line_key in _keys_for_item(item["name"]):
                _cart().pop(line_key, None)
            _set_pending(False)
            details, total = _cart_details()

            return {
                "removed": {
                    "name": item["name"],
                    "quantity": current,
                },
                "cart": details,
                "subtotal": total,
            }
            

    if name == "set_item_quantity":
        item = _find_item(args.get("item_name", ""))
        quantity = _parse_quantity(args.get("quantity"), default=0)

        if not item:
            return {
                "error": "Item is not on this store's menu.",
                "available_items": _menu_names(),
            }

        if quantity is None:
            return {
                "error": "Quantity must be a whole number between 1 and 20."
            }

        with _CART_LOCK:
            keys = _keys_for_item(item["name"])
            if not keys:
                details, total = _cart_details()
                return {
                    "error": f"{item['name']} is not in the cart.",
                    "cart": details,
                    "subtotal": total,
                }

            if len(keys) > 1:
                return {"error": f"{item['name']} has multiple customization lines; remove one line or replace the item."}
            previous_quantity = _cart()[keys[0]]
            _cart()[keys[0]] = quantity
            _set_pending(False)
            details, total = _cart_details()

            return {
                "updated": {
                    "name": item["name"],
                    "previous_quantity": previous_quantity,
                    "quantity": quantity,
                },
                "cart": details,
                "subtotal": total,
            }
            
            

    if name == "replace_cart_item":
        old_item = _find_item(args.get("old_item_name", ""))
        new_item = _find_item(args.get("new_item_name", ""))

        # Validate both items before modifying the cart.
        if not old_item:
            return {
                "error": "The item to replace is not on this store's menu.",
                "available_items": _menu_names(),
            }

        if not new_item:
            return {
                "error": "The replacement item is not on this store's menu.",
                "available_items": _menu_names(),
            }

        if old_item["name"] == new_item["name"]:
            return {
                "error": "The old and replacement items are the same."
            }

        with _CART_LOCK:
            old_name = old_item["name"]
            new_name = new_item["name"]
            old_quantity = _quantity_for_item(old_name)

            if old_quantity == 0:
                details, total = _cart_details()
                return {
                    "error": f"{old_name} is not in the cart.",
                    "cart": details,
                    "subtotal": total,
                }

            new_quantity = _quantity_for_item(new_name) + old_quantity

            if new_quantity > 20:
                return {
                    "error": (
                        "Replacement would exceed the maximum quantity "
                        "of 20 for the new item. The cart was not changed."
                    ),
                    "cart": _cart_details()[0],
                }

            # Both changes occur under the same lock.
            for line_key in _keys_for_item(old_name):
                _cart().pop(line_key, None)
            _cart()[new_name] = _cart().get(new_name, 0) + old_quantity
            _set_pending(False)
            details, total = _cart_details()

            return {
                "replaced": {
                    "old_item": old_name,
                    "new_item": new_name,
                    "quantity_moved": old_quantity,
                    "new_total_quantity": new_quantity,
                },
                "cart": details,
                "subtotal": total,
            }
    
    
    
    
    if name == "get_cart":
        with _CART_LOCK:
            details, total = _cart_details()
            _set_pending(bool(details))
            return {
                "items": details,
                "subtotal": total,
                "ready_for_confirmation": _pending(),
                "next_step": (
                    "Read every item, quantity, and the subtotal back to the customer, then ask them to "
                    "explicitly confirm before calling create_mock_order."
                    if details
                    else "The cart is empty. Ask the customer what they would like to add."
                ),
            }

    if name == "create_mock_order":
        confirmation = args.get("customer_confirmation", "")
        actual_user_message = _CURRENT_USER_MESSAGE.get()
        if actual_user_message is not None:
            if _normalize(confirmation) != _normalize(actual_user_message):
                return {
                    "error": (
                        "Order not created. customer_confirmation must contain the customer's exact "
                        "latest message."
                    )
                }
            if not _is_explicit_confirmation(actual_user_message):
                return {
                    "error": "Order not created. The customer's latest message is not an explicit confirmation."
                }
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
            if not _pending():
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
        return receipt

    return {"error": f"Unknown tool: {name}"}


async def _execute_tool_impl(
    name: str, arguments: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run a tool against PostgreSQL when configured, otherwise use local memory."""

    import postgres_store

    if postgres_store.configured() and not postgres_store.is_open():
        async with _DATABASE_TOOL_LOCK:
            if not postgres_store.is_open():
                await postgres_store.open_store(MENU)
    state = _session_state()
    if not postgres_store.is_open():
        menu_token = None
        if state.store_id == "sanpeggios-demo":
            pass
        else:
            menu_token = _ACTIVE_MENU.set([])
        try:
            result = await _execute_tool_memory(name, arguments)
            if name == "create_mock_order" and result.get("status") == "MOCK_ORDER_CREATED":
                state.cart.clear()
                _set_pending(False)
            return result
        finally:
            if menu_token is not None:
                _ACTIVE_MENU.reset(menu_token)
    menu_token = None
    if postgres_store.is_open():
        store_id = state.store_id
        cached = _STORE_MENU_CACHE.get(store_id)
        if cached is None or __import__("time").monotonic() - cached[0] > 60:
            db_started = __import__("time").perf_counter()
            try:
                rows = await postgres_store.load_store_menu(store_id)
            finally:
                _log_database_latency("menu_load", db_started)
            if not rows and store_id == "sanpeggios-demo":
                # First-use sync is bounded to a single refresh attempt; retain
                # the validated sample menu if the official site is unavailable.
                import menu_sync
                outcome = await menu_sync.sync_store(store_id)
                if outcome.get("status") == "success":
                    rows = await postgres_store.load_store_menu(store_id)
            if not rows and store_id == "sanpeggios-demo":
                rows = await postgres_store.load_menu()
            _STORE_MENU_CACHE[store_id] = (__import__("time").monotonic(), rows)
            cached = _STORE_MENU_CACHE[store_id]
        menu_token = _ACTIVE_MENU.set(cached[1])

    result = await _execute_tool_memory(name, arguments)
    if name == "create_mock_order" and result.get("status") == "MOCK_ORDER_CREATED":
        confirmation = (arguments or {}).get("customer_confirmation", "")
        if postgres_store.is_open():
            db_started = __import__("time").perf_counter()
            try:
                persisted = await postgres_store.create_order(
                    _ACTIVE_SESSION_ID.get(), result, str(confirmation),
                    store_id=state.store_id,
                    idempotency_key=f"{state.session_id}:{state.reviewed_revision}",
                )
            except Exception as exc:
                result = {"error": f"Order was not saved; your cart is still active. {type(exc).__name__}: {exc}"}
            else:
                state.cart.clear()
                _set_pending(False)
                if persisted.get("order_id") != result.get("order_id"):
                    result = {**result, **persisted, "note": "This order was already saved; returning the existing receipt."}
            finally:
                _log_database_latency("order_commit", db_started)
        else:
            state.cart.clear()
            _set_pending(False)
    if menu_token is not None:
        _ACTIVE_MENU.reset(menu_token)
    return result


async def get_session_cart_snapshot(session_id: str) -> tuple[list[dict[str, Any]], float]:
    """Read a session's cart for UI restoration without enabling order confirmation."""

    import postgres_store
    state = _session_state(session_id)
    menu_token = _ACTIVE_MENU.set(MENU)
    try:
        if postgres_store.is_open():
            db_started = __import__("time").perf_counter()
            try:
                items = await postgres_store.load_store_menu(state.store_id)
            finally:
                _log_database_latency("session_menu_load", db_started)
            if items:
                _ACTIVE_MENU.set(items)
        with _CART_LOCK:
            cart = _cart()
            active_id = _ACTIVE_SESSION_ID.set(session_id)
            try:
                return _cart_details()
            finally:
                _ACTIVE_SESSION_ID.reset(active_id)
    finally:
        _ACTIVE_MENU.reset(menu_token)


async def ensure_persisted_session(session_id: str, store_id: str = "sanpeggios-demo") -> None:
    import postgres_store

    _session_state(session_id)
    if postgres_store.is_open():
        await postgres_store.ensure_session(session_id, store_id)


async def session_welcome(session_id: str, store_id: str = "sanpeggios-demo") -> str | None:
    """Return a store's configured welcome exactly once per in-memory call."""
    state = _session_state(session_id)
    state.store_id = store_id
    if state.welcomed:
        return None
    message = "Thank you for calling Sanpeggio's Pizza! Welcome. I'm your virtual ordering assistant. I can help you explore our menu and place your order. What would you like today?"
    import postgres_store
    if postgres_store.is_open():
        settings = await postgres_store.get_store_settings(store_id)
        if settings:
            message = settings["welcome_message"]
    state.welcomed = True
    return message


async def require_fresh_cart_review(session_id: str) -> None:
    global PENDING_CONFIRMATION
    import postgres_store

    state = _session_state(session_id)
    state.pending_confirmation = False
    state.state = "ACTIVE"
    state.reviewed_revision = -1

async def execute_tool(
    name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute an order tool through LangGraph."""
    from tool_graph import run_tool_graph

    return await run_tool_graph(name, arguments)
