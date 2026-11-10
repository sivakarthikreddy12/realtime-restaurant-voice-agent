# Realtime Restaurant Voice Agent

A local English voice-ordering demo using Groq Cloud for speech recognition,
order parsing, and speech synthesis. The browser UI is served by FastAPI and
keeps its cart and mock order workflow local to the app.

This remains a demo: it never contacts Toast or places a real restaurant order.
Confirmed orders are marked as mock orders and stored locally in PostgreSQL when
configured. Without PostgreSQL, the pre-existing sample menu remains the fallback.

## Run

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
# Edit .env and set GROQ_API_KEY to your key from Groq Console.
python ui_server.py
```

On Windows, use PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env and set GROQ_API_KEY, then:
python ui_server.py
```

Open the URL printed by the server (normally `http://127.0.0.1:8000`). The app
loads `GROQ_API_KEY` from `.env`; `.env` is gitignored. Never commit or share it.
It uses `whisper-large-v3-turbo` for transcription, `openai/gpt-oss-120b` for
strict JSON Schema order parsing, and `canopylabs/orpheus-v1-english` with the
`troy` voice for WAV speech output. Long replies are split into requests of at
most 200 characters and WAV chunks are joined before playback. Headphones help
avoid microphone echo.

## PostgreSQL and menu sync

Start PostgreSQL with Docker, then configure the database connection. Preserve
the Groq key when editing `.env`:

```bash
docker compose up -d postgres
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

## Troubleshooting

The app reports missing keys, transcription/parsing failures, and speech
generation failures in the chat. Check that `.env` contains a valid
`GROQ_API_KEY`, then restart `ui_server.py`. Network errors and Groq API errors
are shown in the app. If the error says `model_terms_required`, an organization
admin must accept the Orpheus model terms in Groq Console before speech can be
generated. Text replies remain available if TTS fails.

Useful options:

```bash
python ui_server.py --ui-port 8001
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m pytest -q
```

`unittest` covers the menu and session cases. `pytest` also collects the async
batch/cart and graph tests.
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
