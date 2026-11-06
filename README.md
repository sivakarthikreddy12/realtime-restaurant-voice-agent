# Realtime Restaurant Voice Agent

A local English voice-ordering demo built on Hugging Face `speech-to-speech`.
Audio flows through VAD → STT → LLM → tools → TTS. The browser UI is served by
FastAPI and bridges the local Realtime backend over WebSockets.

This remains a demo: it never contacts Toast or places a real restaurant order.
Confirmed orders are marked as mock orders and stored locally in PostgreSQL when
configured. Without PostgreSQL, the pre-existing sample menu remains the fallback.

## Run

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --check
python ui_server.py
```

On Windows, use the CPU profile and dedicated environment:

```powershell
docker compose up -d postgres
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv_new\Scripts\Activate.ps1
python -m pip install -r requirements-windows.txt
python main.py --check
python ui_server.py
```

The UI loads `.env` from the project folder before opening PostgreSQL. Existing
PowerShell environment variables take precedence over values in `.env`.

Open the URL printed by the server (normally `http://127.0.0.1:8000`). `Ctrl+C`
stops the UI and the voice backend it started. The first voice launch downloads
the selected models; use `python ui_server.py --small-models` for a smaller
local model set. That preset uses Whisper Small MLX for English transcription.
Headphones help avoid microphone echo.

## PostgreSQL and menu sync

Start PostgreSQL with Docker, then configure the connection:

```bash
docker compose up -d postgres
cp .env.example .env
export DATABASE_URL="$(sed -n 's/^DATABASE_URL=//p' .env)"
export SESSION_TTL_SECONDS="$(sed -n 's/^SESSION_TTL_SECONDS=//p' .env)"
python ui_server.py
```

The app creates/migrates tables at startup and refreshes Sanpeggio's official
menu on every UI startup. A failed fetch is logged and leaves any existing
catalog intact. Refresh a menu while the server is running with:

```bash
curl -X POST 'http://127.0.0.1:8000/menu/sync?store_id=sanpeggios-demo'
```

The importer reads the official Wix page's `data-hook` fields for menu rows,
descriptions, prices, and categories. It currently extracts 66 menu and
option rows from 13 categories. Rows without a published price remain in
the catalog for reference but cannot be added to the cart. Unpriced options are
not assigned guessed prices. If the page structure changes or fewer than ten
rows can be parsed, sync fails safely and retains the previous catalog. The
source URL and latest successful sync time are stored with every catalog row;
`menu_sync_status` records sync attempts and errors.

PostgreSQL schema additions:

- `restaurant_stores`: store identity, menu URL, currency, active flag, welcome text.
- `store_menu_categories` → `store_menu_products` → `store_menu_variants`:
  separates sections, explainable menu items, and size/price choices. Product
  descriptions and aliases stay with the product; published prices stay with
  each variant.
- `store_menu_modifier_groups` → `store_menu_modifiers`: stores choices such as
  crusts and toppings, along with published price rules and size restrictions.
- `store_menu_category_notes`: keeps section notes separate from orderable items.
- `store_menu_items`: flat store-scoped projection retained for deterministic
  cart validation; its `(store_id, item_key)` key makes refreshes idempotent.
- `menu_sync_status`: attempt status, item count, last success and error.
- `customer_sessions.store_id` and `orders.store_id` associate records with a store.
- `orders.idempotency_key` prevents a retry of one reviewed cart from inserting
  a second finalized order. Order header and lines are committed atomically.

The legacy `menu_items`, `carts`, and session columns remain for compatibility
with the original prototype. Active call carts are now held in process memory;
intermediate edits are not written to the database. Confirmed mock orders are
the only cart transactions persisted. The process-local session manager allows
independent concurrent calls, retains disconnected sessions for reconnection,
and expires idle sessions after `SESSION_TTL_SECONDS` (default 7200).

## Menu and cart safety

Menu reads and cart edits use deterministic exact-name/alias matching against
the active store catalog. Every batch row is checked independently. Missing,
unavailable, and unpriced rows are rejected; invalid entries do not silently
substitute another item. The database menu is cached in memory for 60 seconds
per store and refreshed after a successful sync. With PostgreSQL off, the
original built-in sample menu is used as the documented fallback.

Cart state follows `ACTIVE → REVIEWED → CONFIRMED`. Every successful edit
invalidates review; `get_cart` returns the authoritative item lines and
subtotal. Checkout requires a fresh review plus an explicit affirmative tied to
the latest customer message. The cart is cleared only after the PostgreSQL
transaction succeeds. A failed write returns an error and leaves the cart
available for retry. Session IDs are UUIDs stored in per-tab session storage;
`?store_id=...` selects a store for the connection. Configure other locations
with `postgres_store.configure_store(store_id, name, menu_url, welcome_message)`;
each store then gets its own catalog, menu cache, session association, and welcome.

The new-call welcome is configurable in `restaurant_stores.welcome_message`
and is played once for a new in-memory session. The default Sanpeggio's message
is seeded on first database setup.

## Latency and diagnostics

The WebSocket bridge writes `[LATENCY]` entries to its terminal and
`latency.log` for speech start/end, transcription completion, response start,
first text, first audio, response completion, tool completion, and database
menu/order operations. Browser `audio_end` currently appends 750 ms of silence
to trigger the configured 700 ms VAD pause (previously 1.2 seconds). The voice
library's model loading, LLM first-token generation, and audio device buffering
remain the main machine-dependent bottlenecks. A trustworthy before/after
end-to-end benchmark requires running the same cached models and hardware; no
baseline trace is included in this repository.

Useful options:

```bash
python main.py --check
python ui_server.py --small-models
python ui_server.py --s2s-port 8766 --ui-port 8001
tail -f latency.log
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m pytest -q
```

`unittest` covers the launcher and Realtime contract plus the new menu and
session cases. `pytest` also collects the async batch/cart and graph tests.
PostgreSQL integration tests require a running local database; the unit suites
do not create or reset one.

## Scope

- The Sanpeggio page does not publish a price for every row; missing prices are
  intentionally kept unorderable until an authoritative price is available.
- The local fallback remains the original sample menu when PostgreSQL/menu sync
  is unavailable.
- Only the listed build-your-own pizza crust/topping options carry parsed
  published surcharges; other menu modifiers remain unorderable until synced
  with authoritative prices. This prototype has no payment, inventory, tax,
  or real POS integration. Bind the UI to localhost only.
- Model-directed tool calling remains imperfect; deterministic server checks
  protect menu lookup, cart state, and mock checkout.
