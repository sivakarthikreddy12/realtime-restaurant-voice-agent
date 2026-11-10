"""PostgreSQL persistence for store catalogs and finalized demo orders."""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any

_pool: Any = None
_disabled_for_process = False


def configured() -> bool:
    return bool(os.environ.get("DATABASE_URL")) and not _disabled_for_process


def mark_unavailable() -> None:
    """Use the in-memory demo store after an unsuccessful startup connection."""
    global _disabled_for_process
    _disabled_for_process = True


def is_open() -> bool:
    return _pool is not None


async def open_store(seed_menu: list[dict[str, Any]]) -> bool:
    """Open the pool, create tables, and seed the menu on first run."""

    global _pool, _disabled_for_process
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return False
    try:
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise RuntimeError(
            "PostgreSQL is configured but psycopg is missing. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    _pool = AsyncConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=5,
        kwargs={"row_factory": dict_row},
        open=False,
    )
    try:
        await _pool.open()
        await _pool.wait()

        statements = (
        """CREATE TABLE IF NOT EXISTS menu_items (
            name TEXT PRIMARY KEY,
            price NUMERIC(10, 2) NOT NULL CHECK (price >= 0),
            aliases TEXT[] NOT NULL DEFAULT '{}',
            active BOOLEAN NOT NULL DEFAULT TRUE,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""",
        """CREATE TABLE IF NOT EXISTS customer_sessions (
            session_id UUID PRIMARY KEY,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            pending_confirmation BOOLEAN NOT NULL DEFAULT FALSE
        )""",
        """CREATE TABLE IF NOT EXISTS carts (
            session_id UUID NOT NULL REFERENCES customer_sessions(session_id) ON DELETE CASCADE,
            item_name TEXT NOT NULL REFERENCES menu_items(name),
            quantity INTEGER NOT NULL CHECK (quantity BETWEEN 1 AND 20),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (session_id, item_name)
        )""",
        """CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            session_id UUID NOT NULL REFERENCES customer_sessions(session_id),
            status TEXT NOT NULL,
            subtotal NUMERIC(10, 2) NOT NULL,
            currency CHAR(3) NOT NULL DEFAULT 'USD',
            customer_confirmation TEXT NOT NULL,
            real_order_placed BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""",
        "CREATE INDEX IF NOT EXISTS orders_session_created_idx ON orders(session_id, created_at DESC)",
        """CREATE TABLE IF NOT EXISTS restaurant_stores (
            store_id TEXT PRIMARY KEY, name TEXT NOT NULL, menu_source_url TEXT,
            welcome_message TEXT NOT NULL, currency CHAR(3) NOT NULL DEFAULT 'USD',
            active BOOLEAN NOT NULL DEFAULT TRUE, updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""",
        """CREATE TABLE IF NOT EXISTS store_menu_items (
            store_id TEXT NOT NULL REFERENCES restaurant_stores(store_id) ON DELETE CASCADE,
            item_key TEXT NOT NULL, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL DEFAULT '', price NUMERIC(10,2) CHECK(price IS NULL OR price >= 0),
            size TEXT, aliases TEXT[] NOT NULL DEFAULT '{}', customizations JSONB NOT NULL DEFAULT '{}'::jsonb,
            available BOOLEAN NOT NULL DEFAULT TRUE, source_url TEXT NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now(), active BOOLEAN NOT NULL DEFAULT TRUE,
            PRIMARY KEY(store_id, item_key)
        )""",
        "CREATE INDEX IF NOT EXISTS store_menu_active_category_idx ON store_menu_items(store_id, active, category)",
        "CREATE INDEX IF NOT EXISTS store_menu_name_idx ON store_menu_items(store_id, lower(name))",
        "ALTER TABLE store_menu_items ALTER COLUMN price DROP NOT NULL",
        """CREATE TABLE IF NOT EXISTS store_menu_categories (
            category_id BIGSERIAL PRIMARY KEY,
            store_id TEXT NOT NULL REFERENCES restaurant_stores(store_id) ON DELETE CASCADE,
            category_key TEXT NOT NULL, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
            display_order INTEGER NOT NULL DEFAULT 0, active BOOLEAN NOT NULL DEFAULT TRUE,
            source_url TEXT NOT NULL, synced_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE(store_id,category_key)
        )""",
        """CREATE TABLE IF NOT EXISTS store_menu_products (
            product_id BIGSERIAL PRIMARY KEY,
            store_id TEXT NOT NULL REFERENCES restaurant_stores(store_id) ON DELETE CASCADE,
            category_id BIGINT NOT NULL REFERENCES store_menu_categories(category_id) ON DELETE CASCADE,
            product_key TEXT NOT NULL, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
            aliases TEXT[] NOT NULL DEFAULT '{}', entry_type TEXT NOT NULL DEFAULT 'product',
            available BOOLEAN NOT NULL DEFAULT TRUE, display_order INTEGER NOT NULL DEFAULT 0,
            active BOOLEAN NOT NULL DEFAULT TRUE, source_url TEXT NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now(), UNIQUE(store_id,product_key)
        )""",
        """CREATE TABLE IF NOT EXISTS store_menu_variants (
            variant_id BIGSERIAL PRIMARY KEY,
            product_id BIGINT NOT NULL REFERENCES store_menu_products(product_id) ON DELETE CASCADE,
            variant_key TEXT NOT NULL, name TEXT NOT NULL, size TEXT,
            price NUMERIC(10,2) CHECK(price IS NULL OR price >= 0),
            price_status TEXT NOT NULL DEFAULT 'not_listed',
            available BOOLEAN NOT NULL DEFAULT TRUE, display_order INTEGER NOT NULL DEFAULT 0,
            UNIQUE(product_id,variant_key)
        )""",
        """CREATE TABLE IF NOT EXISTS store_menu_modifier_groups (
            group_id BIGSERIAL PRIMARY KEY,
            product_id BIGINT NOT NULL REFERENCES store_menu_products(product_id) ON DELETE CASCADE,
            group_key TEXT NOT NULL, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
            required BOOLEAN NOT NULL DEFAULT FALSE, min_select INTEGER NOT NULL DEFAULT 0,
            max_select INTEGER, pricing_rules JSONB NOT NULL DEFAULT '{}'::jsonb,
            constraints JSONB NOT NULL DEFAULT '{}'::jsonb,
            display_order INTEGER NOT NULL DEFAULT 0, UNIQUE(product_id,group_key)
        )""",
        """CREATE TABLE IF NOT EXISTS store_menu_modifiers (
            modifier_id BIGSERIAL PRIMARY KEY,
            group_id BIGINT NOT NULL REFERENCES store_menu_modifier_groups(group_id) ON DELETE CASCADE,
            modifier_key TEXT NOT NULL, name TEXT NOT NULL,
            price_delta NUMERIC(10,2),
            valid_sizes TEXT[] NOT NULL DEFAULT '{}', display_order INTEGER NOT NULL DEFAULT 0,
            UNIQUE(group_id,modifier_key)
        )""",
        "ALTER TABLE store_menu_modifiers DROP CONSTRAINT IF EXISTS store_menu_modifiers_price_delta_check",
        """CREATE TABLE IF NOT EXISTS store_menu_category_notes (
            note_id BIGSERIAL PRIMARY KEY,
            category_id BIGINT NOT NULL REFERENCES store_menu_categories(category_id) ON DELETE CASCADE,
            note_key TEXT NOT NULL, title TEXT NOT NULL DEFAULT '', text TEXT NOT NULL,
            display_order INTEGER NOT NULL DEFAULT 0, UNIQUE(category_id,note_key)
        )""",
        "CREATE INDEX IF NOT EXISTS store_menu_products_category_idx ON store_menu_products(category_id,active,display_order)",
        "CREATE INDEX IF NOT EXISTS store_menu_variants_product_idx ON store_menu_variants(product_id,display_order)",
        "ALTER TABLE customer_sessions ADD COLUMN IF NOT EXISTS store_id TEXT NOT NULL DEFAULT 'sanpeggios-demo'",
        "ALTER TABLE orders ADD COLUMN IF NOT EXISTS store_id TEXT NOT NULL DEFAULT 'sanpeggios-demo'",
        "ALTER TABLE orders ADD COLUMN IF NOT EXISTS idempotency_key TEXT",
        "CREATE UNIQUE INDEX IF NOT EXISTS orders_session_idempotency_idx ON orders(session_id,idempotency_key) WHERE idempotency_key IS NOT NULL",
        """CREATE TABLE IF NOT EXISTS menu_sync_status (
            store_id TEXT PRIMARY KEY REFERENCES restaurant_stores(store_id) ON DELETE CASCADE,
            source_url TEXT NOT NULL, last_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_success_at TIMESTAMPTZ, item_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending', error TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS order_items (
            id BIGSERIAL PRIMARY KEY,
            order_id TEXT NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
            item_name TEXT NOT NULL,
            quantity INTEGER NOT NULL CHECK (quantity > 0),
            unit_price NUMERIC(10, 2) NOT NULL,
            line_total NUMERIC(10, 2) NOT NULL,
            customizations JSONB NOT NULL DEFAULT '{}'::jsonb
        )""",
        "ALTER TABLE order_items ADD COLUMN IF NOT EXISTS customizations JSONB NOT NULL DEFAULT '{}'::jsonb",
        )
        async with _pool.connection() as conn:
            for statement in statements:
                await conn.execute(statement)
            await conn.execute(
                """INSERT INTO restaurant_stores(store_id,name,menu_source_url,welcome_message)
                   VALUES ('sanpeggios-demo', %s, %s, %s) ON CONFLICT(store_id) DO NOTHING""",
                ("Sanpeggio's Pizza", "https://www.sanpeggiopizza.com/menu?menu=mainmenu",
                 "Thank you for calling Sanpeggio's Pizza! Welcome. I'm your virtual ordering assistant. I can help you explore our menu and place your order. What would you like today?"),
            )
            count = await (await conn.execute("SELECT count(*) AS count FROM menu_items")).fetchone()
            if count["count"] == 0:
                for item in seed_menu:
                    await conn.execute(
                        "INSERT INTO menu_items (name, price, aliases) VALUES (%s, %s, %s)",
                        (item["name"], item["price"], item.get("aliases", [])),
                    )
    except Exception:
        await _pool.close()
        _pool = None
        raise
    _disabled_for_process = False
    return True


async def close_store() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def _require_pool() -> Any:
    if _pool is None:
        raise RuntimeError("PostgreSQL store is not open. Start the application lifespan first.")
    return _pool


async def load_menu() -> list[dict[str, Any]]:
    pool = _require_pool()
    async with pool.connection() as conn:
        rows = await (
            await conn.execute(
                "SELECT name, price, aliases FROM menu_items WHERE active = TRUE ORDER BY name"
            )
        ).fetchall()
    return [
        {"name": row["name"], "price": float(row["price"]), "aliases": list(row["aliases"])}
        for row in rows
    ]


def _catalog_key(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _catalog_source_groups(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn scraped rows into category → product → variant/modifier groups."""
    import re

    categories: list[dict[str, Any]] = []
    by_category: dict[str, dict[str, Any]] = {}
    for item in items:
        category_name = str(item.get("category") or "Menu").strip()
        category_key = _catalog_key(category_name)
        category = by_category.get(category_key)
        if category is None:
            category = {"key": category_key, "name": category_name, "products": [], "notes": [], "_products": {}}
            by_category[category_key] = category
            categories.append(category)

    option_names: set[str] = set()
    for item in items:
        for group in (item.get("customizations") or {}).values():
            option_names.update(_catalog_key(name) for name in group.get("choices", []))

    for position, item in enumerate(items):
        category_name = str(item.get("category") or "Menu").strip()
        category = by_category[_catalog_key(category_name)]
        name = str(item.get("name") or "").strip()
        description = str(item.get("description") or "").strip()
        if name == "-":
            if description:
                category["notes"].append({"title": "", "text": description, "position": position})
            continue
        if _catalog_key(name) in option_names:
            continue
        if name.casefold() in {"choose your size", "choose your crust", "choose your toppings", "specialty & gourmet"}:
            continue

        variant_name = str(item.get("size") or "").strip()
        product_name = name
        match = re.match(r"^(Build Your Own Pizza|Specialty/Gourmet Pizza)\s*-\s*(.+)$", name, re.I)
        if match:
            product_name, variant_name = match.group(1), match.group(2).strip()

        product_key = f"{category['key']}:{_catalog_key(product_name)}"
        product = category["_products"].get(product_key)
        if product is None:
            product = {
                "key": product_key, "name": product_name,
                "description": description, "aliases": [], "customizations": {},
                "variants": [], "position": position,
                "available": bool(item.get("available", True)),
            }
            category["_products"][product_key] = product
            category["products"].append(product)
        elif description and not product["description"]:
            product["description"] = description

        product["aliases"].extend(item.get("aliases", []))
        product["available"] = product["available"] and bool(item.get("available", True))
        for group_name, group in (item.get("customizations") or {}).items():
            product["customizations"].setdefault(group_name, group)

        price = item.get("price")
        if price is not None:
            variant_label = variant_name or "Standard"
            variant_key = _catalog_key(variant_label)
            if not any(existing["key"] == variant_key for existing in product["variants"]):
                product["variants"].append({
                    "key": variant_key, "name": variant_label,
                    "size": item.get("size"), "price": price,
                    "available": bool(item.get("available", True)),
                    "position": position,
                })

    for category in categories:
        category["products"].sort(key=lambda product: product["position"])
        for product in category["products"]:
            product["aliases"] = sorted(set(product["aliases"]))
            product["variants"].sort(key=lambda variant: variant["position"])
        del category["_products"]
    return categories


async def upsert_store_menu(
    store_id: str, items: list[dict[str, Any]], *, source_url: str,
) -> None:
    """Replace one store's active snapshot transactionally, retaining stable keys."""
    if not items:
        raise ValueError("Refusing to store an empty menu")
    pool = _require_pool()
    async with pool.connection() as conn:
        async with conn.transaction():
            await conn.execute(
                """INSERT INTO restaurant_stores(store_id,name,menu_source_url,welcome_message)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT(store_id) DO UPDATE SET menu_source_url=EXCLUDED.menu_source_url,
                   updated_at=now()""",
                (store_id, "Sanpeggio's Pizza", source_url,
                 "Thank you for calling Sanpeggio's Pizza! Welcome. I'm your virtual ordering assistant. I can help you explore our menu and place your order. What would you like today?"),
            )
            await conn.execute("UPDATE store_menu_items SET active=FALSE WHERE store_id=%s", (store_id,))
            for item in items:
                name = str(item["name"]).strip()
                key = f"{str(item.get('category', '')).casefold()}:{' '.join(name.casefold().split())}"
                await conn.execute(
                    """INSERT INTO store_menu_items
                       (store_id,item_key,name,description,category,price,size,aliases,customizations,available,source_url,synced_at,active)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,now(),TRUE)
                       ON CONFLICT(store_id,item_key) DO UPDATE SET name=EXCLUDED.name,
                       description=EXCLUDED.description,category=EXCLUDED.category,price=EXCLUDED.price,
                       size=EXCLUDED.size,aliases=EXCLUDED.aliases,customizations=EXCLUDED.customizations,
                       available=EXCLUDED.available,source_url=EXCLUDED.source_url,synced_at=now(),active=TRUE""",
                    (store_id, key, name, item.get("description", ""), item.get("category", ""),
                     item["price"], item.get("size"), item.get("aliases", []),
                     __import__("json").dumps(item.get("customizations", {})),
                     item.get("available", True), source_url),
                )

            # Store an explanation-friendly catalog in normalized tables. Keep
            # store_menu_items above as the flat projection used by cart tools.
            await conn.execute("UPDATE store_menu_categories SET active=FALSE WHERE store_id=%s", (store_id,))
            await conn.execute("UPDATE store_menu_products SET active=FALSE WHERE store_id=%s", (store_id,))
            catalog = _catalog_source_groups(items)
            for category_order, category in enumerate(catalog):
                category_row = await (await conn.execute(
                    """INSERT INTO store_menu_categories
                       (store_id,category_key,name,display_order,active,source_url,synced_at)
                       VALUES (%s,%s,%s,%s,TRUE,%s,now())
                       ON CONFLICT(store_id,category_key) DO UPDATE SET name=EXCLUDED.name,
                       display_order=EXCLUDED.display_order,active=TRUE,source_url=EXCLUDED.source_url,synced_at=now()
                       RETURNING category_id""",
                    (store_id, category["key"], category["name"], category_order, source_url),
                )).fetchone()
                category_id = category_row["category_id"]
                await conn.execute("DELETE FROM store_menu_category_notes WHERE category_id=%s", (category_id,))
                for note_order, note in enumerate(category["notes"]):
                    await conn.execute(
                        """INSERT INTO store_menu_category_notes(category_id,note_key,title,text,display_order)
                           VALUES (%s,%s,%s,%s,%s)""",
                        (category_id, f"note-{note_order}", note.get("title", ""), note["text"], note_order),
                    )

                for product_order, product in enumerate(category["products"]):
                    product_row = await (await conn.execute(
                        """INSERT INTO store_menu_products
                           (store_id,category_id,product_key,name,description,aliases,entry_type,available,
                            display_order,active,source_url,synced_at)
                           VALUES (%s,%s,%s,%s,%s,%s,'product',%s,%s,TRUE,%s,now())
                           ON CONFLICT(store_id,product_key) DO UPDATE SET category_id=EXCLUDED.category_id,
                           name=EXCLUDED.name,description=EXCLUDED.description,aliases=EXCLUDED.aliases,
                           available=EXCLUDED.available,display_order=EXCLUDED.display_order,active=TRUE,
                           source_url=EXCLUDED.source_url,synced_at=now()
                           RETURNING product_id""",
                        (store_id, category_id, product["key"], product["name"], product["description"],
                         product["aliases"], product["available"], product_order, source_url),
                    )).fetchone()
                    product_id = product_row["product_id"]
                    await conn.execute("DELETE FROM store_menu_variants WHERE product_id=%s", (product_id,))
                    await conn.execute("DELETE FROM store_menu_modifier_groups WHERE product_id=%s", (product_id,))
                    variants = product["variants"] or [{
                        "key": "standard", "name": "Standard", "size": None,
                        "price": None, "available": product["available"], "position": product_order,
                    }]
                    for variant_order, variant in enumerate(variants):
                        price = variant.get("price")
                        await conn.execute(
                            """INSERT INTO store_menu_variants
                               (product_id,variant_key,name,size,price,price_status,available,display_order)
                               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                            (product_id, variant["key"], variant["name"], variant.get("size"), price,
                             "listed" if price is not None else "not_listed",
                             variant.get("available", product["available"]), variant_order),
                        )
                    for group_order, (group_name, group) in enumerate(product["customizations"].items()):
                        group_key = _catalog_key(group_name)
                        rules = {key: group[key] for key in ("price_by_size", "price_deltas") if key in group}
                        constraints = {key: group[key] for key in ("size_limits", "min_select", "max_select") if key in group}
                        group_row = await (await conn.execute(
                            """INSERT INTO store_menu_modifier_groups
                               (product_id,group_key,name,description,required,min_select,max_select,
                                pricing_rules,constraints,display_order)
                               VALUES (%s,%s,%s,'',FALSE,%s,%s,%s::jsonb,%s::jsonb,%s)
                               RETURNING group_id""",
                            (product_id, group_key, group_name, int(group.get("min_select", 0)),
                             group.get("max_select"), __import__("json").dumps(rules),
                             __import__("json").dumps(constraints), group_order),
                        )).fetchone()
                        group_id = group_row["group_id"]
                        for choice_order, choice in enumerate(group.get("choices", [])):
                            choice_name = str(choice)
                            delta = group.get("price_deltas", {}).get(choice_name)
                            limits = group.get("size_limits", {}).get(choice_name, [])
                            await conn.execute(
                                """INSERT INTO store_menu_modifiers
                                   (group_id,modifier_key,name,price_delta,valid_sizes,display_order)
                                   VALUES (%s,%s,%s,%s,%s,%s)""",
                                (group_id, _catalog_key(choice_name), choice_name, delta,
                                 [str(size) for size in limits], choice_order),
                            )
            await conn.execute(
                """INSERT INTO menu_sync_status(store_id,source_url,last_attempt_at,last_success_at,item_count,status,error)
                   VALUES (%s,%s,now(),now(),%s,'success',NULL)
                   ON CONFLICT(store_id) DO UPDATE SET source_url=EXCLUDED.source_url,
                   last_attempt_at=now(),last_success_at=now(),item_count=EXCLUDED.item_count,status='success',error=NULL""",
                (store_id, source_url, len(items)),
            )


async def record_menu_sync_failure(store_id: str, source_url: str, error: str) -> None:
    pool = _require_pool()
    async with pool.connection() as conn:
        async with conn.transaction():
            await conn.execute(
                """INSERT INTO restaurant_stores(store_id,name,menu_source_url,welcome_message)
                   VALUES (%s,%s,%s,%s) ON CONFLICT(store_id) DO NOTHING""",
                (store_id, "Sanpeggio's Pizza", source_url, "Welcome to Sanpeggio's Pizza! How can I help?"),
            )
            await conn.execute(
                """INSERT INTO menu_sync_status(store_id,source_url,last_attempt_at,status,error)
                   VALUES (%s,%s,now(),'failed',%s)
                   ON CONFLICT(store_id) DO UPDATE SET source_url=EXCLUDED.source_url,
                   last_attempt_at=now(),status='failed',error=EXCLUDED.error""",
                (store_id, source_url, error[:2000]),
            )


async def load_store_menu(store_id: str = "sanpeggios-demo") -> list[dict[str, Any]]:
    pool = _require_pool()
    async with pool.connection() as conn:
        rows = await (await conn.execute(
            """SELECT name,description,category,price,size,aliases,customizations,available,source_url,synced_at
               FROM store_menu_items WHERE store_id=%s AND active=TRUE ORDER BY category,name""", (store_id,)
        )).fetchall()
    return [{**row, "price": float(row["price"]) if row["price"] is not None else None, "aliases": list(row["aliases"]),
             "customizations": row["customizations"], "synced_at": row["synced_at"].isoformat()}
            for row in rows]


async def load_store_menu_catalog(store_id: str = "sanpeggios-demo") -> dict[str, Any]:
    """Load a nested catalog shaped for clear LLM menu explanations."""
    pool = _require_pool()
    async with pool.connection() as conn:
        store = await (await conn.execute(
            """SELECT store_id,name,currency,menu_source_url FROM restaurant_stores
               WHERE store_id=%s AND active=TRUE""", (store_id,)
        )).fetchone()
        if store is None:
            return {"store_id": store_id, "categories": []}
        categories = await (await conn.execute(
            """SELECT category_id,name,description,display_order
               FROM store_menu_categories WHERE store_id=%s AND active=TRUE
               ORDER BY display_order,category_id""", (store_id,)
        )).fetchall()
        products = await (await conn.execute(
            """SELECT product_id,category_id,name,description,aliases,available,display_order
               FROM store_menu_products WHERE store_id=%s AND active=TRUE
               ORDER BY category_id,display_order,product_id""", (store_id,)
        )).fetchall()
        variants = await (await conn.execute(
            """SELECT v.product_id,v.name,v.size,v.price,v.price_status,v.available
               FROM store_menu_variants v JOIN store_menu_products p USING(product_id)
               WHERE p.store_id=%s AND p.active=TRUE
               ORDER BY v.product_id,v.display_order,v.variant_id""", (store_id,)
        )).fetchall()
        groups = await (await conn.execute(
            """SELECT g.group_id,g.product_id,g.name,g.description,g.required,g.min_select,g.max_select,
                      g.pricing_rules,g.constraints
               FROM store_menu_modifier_groups g JOIN store_menu_products p USING(product_id)
               WHERE p.store_id=%s AND p.active=TRUE
               ORDER BY g.product_id,g.display_order,g.group_id""", (store_id,)
        )).fetchall()
        modifiers = await (await conn.execute(
            """SELECT m.group_id,m.name,m.price_delta,m.valid_sizes
               FROM store_menu_modifiers m JOIN store_menu_modifier_groups g USING(group_id)
               JOIN store_menu_products p USING(product_id)
               WHERE p.store_id=%s AND p.active=TRUE
               ORDER BY m.group_id,m.display_order,m.modifier_id""", (store_id,)
        )).fetchall()
        notes = await (await conn.execute(
            """SELECT n.category_id,n.title,n.text FROM store_menu_category_notes n
               JOIN store_menu_categories c USING(category_id)
               WHERE c.store_id=%s AND c.active=TRUE ORDER BY n.category_id,n.display_order,n.note_id""",
            (store_id,),
        )).fetchall()
        sync = await (await conn.execute(
            "SELECT last_success_at FROM menu_sync_status WHERE store_id=%s", (store_id,)
        )).fetchone()

    categories_by_id: dict[int, dict[str, Any]] = {}
    for row in categories:
        category = {"name": row["name"], "description": row["description"], "items": [], "notes": []}
        categories_by_id[row["category_id"]] = category
    products_by_id: dict[int, dict[str, Any]] = {}
    for row in products:
        product = {
            "name": row["name"], "description": row["description"],
            "aliases": list(row["aliases"]), "available": row["available"],
            "price_status": "not_listed", "variants": [], "options": [],
        }
        products_by_id[row["product_id"]] = product
        category = categories_by_id.get(row["category_id"])
        if category is not None:
            category["items"].append(product)
    for row in variants:
        product = products_by_id.get(row["product_id"])
        if product is None:
            continue
        price = float(row["price"]) if row["price"] is not None else None
        product["variants"].append({
            "name": row["name"], "size": row["size"], "price": price,
            "price_status": row["price_status"], "available": row["available"],
        })
        if price is not None:
            product["price_status"] = "listed"
    options_by_group: dict[int, list[dict[str, Any]]] = {}
    for row in modifiers:
        options_by_group.setdefault(row["group_id"], []).append({
            "name": row["name"],
            "price_delta": float(row["price_delta"]) if row["price_delta"] is not None else None,
            "valid_sizes": list(row["valid_sizes"]),
        })
    for row in groups:
        product = products_by_id.get(row["product_id"])
        if product is None:
            continue
        rules = row["pricing_rules"] or {}
        choices = options_by_group.get(row["group_id"], [])
        for choice in choices:
            if rules.get("price_by_size"):
                choice["price_by_size"] = rules["price_by_size"]
        product["options"].append({
            "name": row["name"], "description": row["description"],
            "required": row["required"], "min_select": row["min_select"],
            "max_select": row["max_select"], "choices": choices,
            "constraints": row["constraints"],
        })
    for row in notes:
        category = categories_by_id.get(row["category_id"])
        if category is not None:
            category["notes"].append({"title": row["title"], "text": row["text"]})
    return {
        "store": {"id": store["store_id"], "name": store["name"]},
        "currency": store["currency"].strip(), "source_url": store["menu_source_url"],
        "synced_at": sync["last_success_at"].isoformat() if sync and sync["last_success_at"] else None,
        "categories": list(categories_by_id.values()),
    }


async def get_store_settings(store_id: str = "sanpeggios-demo") -> dict[str, Any] | None:
    pool = _require_pool()
    async with pool.connection() as conn:
        return await (await conn.execute(
            "SELECT store_id,name,welcome_message,currency,menu_source_url FROM restaurant_stores WHERE store_id=%s AND active=TRUE",
            (store_id,),
        )).fetchone()


async def configure_store(
    store_id: str, name: str, menu_source_url: str, welcome_message: str,
    currency: str = "USD",
) -> None:
    """Create/update per-location settings without duplicating order logic."""
    pool = _require_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """INSERT INTO restaurant_stores(store_id,name,menu_source_url,welcome_message,currency)
               VALUES (%s,%s,%s,%s,%s)
               ON CONFLICT(store_id) DO UPDATE SET name=EXCLUDED.name,
               menu_source_url=EXCLUDED.menu_source_url,welcome_message=EXCLUDED.welcome_message,
               currency=EXCLUDED.currency,updated_at=now()""",
            (store_id, name, menu_source_url, welcome_message, currency),
        )


async def ensure_session(session_id: str, store_id: str = "sanpeggios-demo") -> None:
    pool = _require_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """INSERT INTO customer_sessions (session_id,store_id) VALUES (%s::uuid,%s)
               ON CONFLICT (session_id) DO UPDATE SET last_seen_at = now(),store_id=EXCLUDED.store_id""",
            (session_id, store_id),
        )


async def require_cart_review(session_id: str) -> None:
    """Require a fresh read-back after the browser reconnects."""

    pool = _require_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE customer_sessions SET pending_confirmation = FALSE, last_seen_at = now() WHERE session_id = %s::uuid",
            (session_id,),
        )


async def invalidate_confirmation(session_id: str) -> None:
    pool = _require_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE customer_sessions SET pending_confirmation = FALSE, last_seen_at = now() WHERE session_id = %s::uuid",
            (session_id,),
        )


async def load_session(session_id: str) -> tuple[dict[str, int], bool]:
    pool = _require_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """INSERT INTO customer_sessions (session_id) VALUES (%s::uuid)
               ON CONFLICT (session_id) DO UPDATE SET last_seen_at = now()""",
            (session_id,),
        )
        session = await (
            await conn.execute(
                "SELECT pending_confirmation FROM customer_sessions WHERE session_id = %s::uuid",
                (session_id,),
            )
        ).fetchone()
        rows = await (
            await conn.execute(
                "SELECT item_name, quantity FROM carts WHERE session_id = %s::uuid",
                (session_id,),
            )
        ).fetchall()
    return ({row["item_name"]: row["quantity"] for row in rows}, session["pending_confirmation"])


async def save_session(
    session_id: str, cart: dict[str, int], pending_confirmation: bool
) -> None:
    pool = _require_pool()
    async with pool.connection() as conn:
        async with conn.transaction():
            await conn.execute(
                """INSERT INTO customer_sessions (session_id, pending_confirmation)
                   VALUES (%s::uuid, %s)
                   ON CONFLICT (session_id) DO UPDATE
                   SET last_seen_at = now(), pending_confirmation = EXCLUDED.pending_confirmation""",
                (session_id, pending_confirmation),
            )
            await conn.execute("DELETE FROM carts WHERE session_id = %s::uuid", (session_id,))
            for item_name, quantity in cart.items():
                await conn.execute(
                    "INSERT INTO carts (session_id, item_name, quantity) VALUES (%s::uuid, %s, %s)",
                    (session_id, item_name, quantity),
                )


async def create_order(
    session_id: str, receipt: dict[str, Any], confirmation: str,
    *, store_id: str = "sanpeggios-demo", idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Persist an order and all lines atomically; return an existing retry receipt."""

    pool = _require_pool()
    async with pool.connection() as conn:
        async with conn.transaction():
            await conn.execute(
                """INSERT INTO orders
                   (order_id, session_id, store_id, idempotency_key, status, subtotal, currency, customer_confirmation, real_order_placed, created_at)
                   VALUES (%s, %s::uuid, %s, %s, %s, %s, %s, %s, FALSE, %s)
                   ON CONFLICT (session_id,idempotency_key) WHERE idempotency_key IS NOT NULL DO NOTHING""",
                (
                    receipt["order_id"], session_id, store_id, idempotency_key, receipt["status"], receipt["subtotal"],
                    receipt["currency"], confirmation, receipt["created_at"],
                ),
            )
            inserted = await (await conn.execute("SELECT order_id FROM orders WHERE session_id=%s::uuid AND idempotency_key=%s", (session_id, idempotency_key))).fetchone() if idempotency_key else {"order_id": receipt["order_id"]}
            persisted_id = inserted["order_id"]
            if persisted_id != receipt["order_id"]:
                return await _load_order_receipt(conn, persisted_id)
            for item in receipt["items"]:
                await conn.execute(
                    """INSERT INTO order_items (order_id, item_name, quantity, unit_price, line_total, customizations)
                       VALUES (%s, %s, %s, %s, %s, %s::jsonb)""",
                    (
                        receipt["order_id"], item["name"], item["quantity"],
                        item["unit_price"], item["line_total"],
                        __import__("json").dumps(item.get("customizations", {})),
                    ),
                )
            await conn.execute(
                "UPDATE customer_sessions SET pending_confirmation = FALSE, last_seen_at = now() WHERE session_id = %s::uuid",
                (session_id,),
            )
    return receipt


async def _load_order_receipt(conn: Any, order_id: str) -> dict[str, Any]:
    header = await (await conn.execute("SELECT order_id,status,subtotal,currency,real_order_placed,created_at FROM orders WHERE order_id=%s", (order_id,))).fetchone()
    lines = await (await conn.execute("SELECT item_name AS name,quantity,unit_price,line_total,customizations FROM order_items WHERE order_id=%s ORDER BY id", (order_id,))).fetchall()
    return {**header, "subtotal": float(header["subtotal"]),
            "created_at": header["created_at"].isoformat(), "items": [
        {**line, "unit_price": float(line["unit_price"]), "line_total": float(line["line_total"])} for line in lines
    ]}


async def list_orders(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Read recent persisted orders for a session (available to app/admin code)."""

    pool = _require_pool()
    async with pool.connection() as conn:
        rows = await (
            await conn.execute(
                """SELECT order_id, status, subtotal, currency, real_order_placed, created_at
                   FROM orders WHERE session_id = %s::uuid ORDER BY created_at DESC LIMIT %s""",
                (session_id, limit),
            )
        ).fetchall()
    return [
        {
            **row,
            "subtotal": float(Decimal(row["subtotal"])),
            "created_at": row["created_at"].isoformat(),
        }
        for row in rows
    ]
