"""Sanpeggio's published-menu importer.

The restaurant page is rendered by Wix. Its server HTML currently includes
``data-hook`` attributes for menu categories and rows, so this parser avoids
depending on Wix's generated CSS class names.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

SOURCE_URL = "https://www.sanpeggiopizza.com/menu?menu=mainmenu"
STORE_ID = "sanpeggios-demo"
log = logging.getLogger(__name__)


def _money(value: str) -> float | None:
    match = re.search(r"\$\s*(\d+(?:\.\d{1,2})?)", value)
    return float(match.group(1)) if match else None


class _MenuParser(HTMLParser):
    """Read only explicitly marked menu rows from the page HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.category = ""
        self.rows: list[dict[str, Any]] = []
        self.row: dict[str, Any] | None = None
        self.capture: str | None = None
        self.capture_tag: str | None = None
        self.row_stack: list[str] = []
        self.buffer: list[str] = []
        self.heading_depth = 0
        self.heading: list[str] = []
        self.heading_previous_category = ""
        self.last_heading = ""
        self.last_heading_category = ""
        self.section_description = False
        self.section_buffer: list[str] = []
        self.section_stack: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = dict(attrs)
        hook = attrs_map.get("data-hook", "")
        void_tags = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
        if self.section_stack and tag not in void_tags:
            self.section_stack[-1]["depth"] += 1
        if hook == "section.container":
            self.section_stack.append({"name": "", "category": self.category, "description": "", "item_count": 0, "depth": 1})
        if re.fullmatch(r"h[1-6]", tag):
            if not self.heading_depth:
                self.heading_previous_category = self.category
            self.heading_depth += 1
        if attrs_map.get("data-hook") == "section.description":
            self.section_description = True
            self.section_buffer = []
        if hook == "item.root":
            if self.section_stack:
                self.section_stack[-1]["item_count"] += 1
            self.row = {"category": self.category, "name": "", "description": "", "price": None}
            self.row_stack = [tag]
        elif self.row is not None:
            if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
                self.row_stack.append(tag)
            if hook in {"item.name", "item.description", "item.price"}:
                self.capture = hook.rsplit(".", 1)[1]
                self.capture_tag = tag
                self.buffer = []

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if re.fullmatch(r"h[1-6]", tag) and self.heading_depth:
            self.heading_depth -= 1
            if not self.heading_depth and self.heading:
                heading = " ".join("".join(self.heading).split())
                if heading:
                    self.last_heading = heading
                    self.last_heading_category = self.heading_previous_category or heading
                    self.category = heading
                    if self.section_stack:
                        self.section_stack[-1]["name"] = heading
                self.heading.clear()
        if self.section_description and tag == "p":
            description = " ".join("".join(self.section_buffer).split())
            if description and self.section_stack:
                self.section_stack[-1]["description"] = description
            self.section_description = False
            self.section_buffer = []
        if self.row is not None:
            if self.capture and tag == self.capture_tag:
                value = " ".join("".join(self.buffer).split())
                if self.capture == "price":
                    self.row["price"] = _money(value)
                else:
                    self.row[self.capture] = value
                self.capture, self.capture_tag, self.buffer = None, None, []
            if tag in self.row_stack:
                while self.row_stack:
                    popped = self.row_stack.pop()
                    if popped == tag:
                        break
            if not self.row_stack:
                if self.row.get("name"):
                    self.rows.append(self.row)
                self.row = None
        if self.section_stack:
            self.section_stack[-1]["depth"] -= 1
            if self.section_stack[-1]["depth"] <= 0:
                section = self.section_stack.pop()
                if section["item_count"] == 0 and section["name"] and section["description"]:
                    self.rows.append({"category": section["category"], "name": section["name"],
                                      "description": section["description"], "price": None})

    def handle_data(self, data: str) -> None:
        if self.heading_depth:
            self.heading.append(data)
        if self.section_description:
            self.section_buffer.append(data)
        if self.capture:
            self.buffer.append(data)


def parse_menu_html(html: str, *, source_url: str = SOURCE_URL) -> list[dict[str, Any]]:
    parser = _MenuParser()
    parser.feed(html)
    # Fail closed: never interpret a partially loaded/changed page as a menu.
    if len(parser.rows) < 10:
        raise ValueError(f"Only {len(parser.rows)} priced menu rows found; expected at least 10")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    homestyle = [row for row in parser.rows if row["category"] == "Homestyle Pizza"]
    crusts = [row["name"] for row in homestyle if row["name"] in {"New York Style", "Thin & Crispy"}]
    if any("Cauliflower Crust" in row["name"] for row in homestyle):
        crusts.append("Cauliflower Crust")
    toppings_row = next((row for row in homestyle if row["name"] == "Choose Your Toppings"), None)
    topping_options: list[str] = []
    topping_price_by_size: dict[str, float] = {}
    if toppings_row:
        desc = toppings_row["description"]
        for label, amount in re.findall(r"(Small|Medium|Large|X-Large)\s*\$\s*(\d+(?:\.\d{1,2})?)", desc, re.I):
            topping_price_by_size[label.casefold()] = float(amount)
        # Toppings follow the last printed size price on the same description.
        tail = re.split(r"X-Large\s*\$\s*\d+(?:\.\d{1,2})?", desc, flags=re.I)
        if len(tail) > 1:
            topping_options = [x.strip() for x in re.split(r"\s*\|\s*", tail[-1]) if x.strip()]
    customization_data = {
        "crust": {"choices": crusts, "price_deltas": {"Thin & Crispy": 2.0, "Cauliflower Crust": 3.0},
                  "size_limits": {"Thin & Crispy": [14, 16], "Cauliflower Crust": [12]}},
        "toppings": {"choices": topping_options, "price_by_size": topping_price_by_size},
    }
    for row in parser.rows:
        name = row["name"]
        aliases: list[str] = []
        customizations: dict[str, Any] = {}
        if row["category"] == "Homestyle Pizza" and row["price"] is not None and re.match(r"^(7|12|14|16)\s*[”\"]", name):
            size_match = re.match(r"^(\d+)\s*[”\"]\s*(.*)$", name)
            size_label = size_match.group(2).strip() if size_match else name
            name = f"Build Your Own Pizza - {name}"
            alias_size = {"7": "small", "12": "medium", "14": "large", "16": "extra large"}.get(size_match.group(1), size_label.casefold()) if size_match else size_label.casefold()
            aliases = [f"{alias_size} pizza", f"{size_match.group(1)} inch pizza"] if size_match else []
            customizations = customization_data
        elif row["category"] == "Neighborhood Favorites" and row["price"] is not None and re.match(r"^(7|12|14|16)\s*[”\"]", name):
            size_match = re.match(r"^(\d+)\s*[”\"]\s*(.*)$", name)
            size_label = size_match.group(2).strip() if size_match else name
            name = f"Specialty/Gourmet Pizza - {name}"
            aliases = [f"{size_label.casefold()} specialty pizza", f"{size_label.casefold()} gourmet pizza"]
        key = (row["category"].casefold(), name.casefold())
        if key in seen:
            continue
        seen.add(key)
        desc = row["description"]
        sizes = re.findall(r"\b(?:7|8|10|12|14|16|18|20)\s*[”\"](?:\s*[A-Za-z-]+)?", name, re.I)
        normalized.append({
            "name": name, "description": desc, "category": row["category"],
        "price": row["price"], "size": sizes[0] if sizes else None,
            "aliases": aliases, "customizations": customizations, "available": True,
            "source_url": source_url,
        })
    return normalized


def fetch_menu(url: str = SOURCE_URL, timeout: float = 20) -> list[dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": "RestaurantMenuSync/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    return parse_menu_html(body.decode(charset, errors="replace"), source_url=url)


async def sync_store(store_id: str = STORE_ID, url: str | None = None) -> dict[str, Any]:
    """Fetch off-loop and atomically upsert a complete menu into PostgreSQL."""
    started = datetime.now(timezone.utc)
    try:
        if url is None:
            import postgres_store
            settings = await postgres_store.get_store_settings(store_id) if postgres_store.is_open() else None
            url = (settings or {}).get("menu_source_url") or SOURCE_URL
        items = await asyncio.to_thread(fetch_menu, url)
        import postgres_store
        await postgres_store.upsert_store_menu(store_id, items, source_url=url)
        result = {"status": "success", "store_id": store_id, "item_count": len(items),
                  "source_url": url, "synced_at": started.isoformat()}
        log.info("Menu sync complete: %s", json.dumps(result))
        print("[MENU_SYNC] " + json.dumps(result), flush=True)
        return result
    except Exception as exc:
        log.exception("Menu sync failed for %s", store_id)
        print(f"[MENU_SYNC] failed store={store_id} error={type(exc).__name__}: {exc}", flush=True)
        try:
            import postgres_store
            if postgres_store.is_open():
                await postgres_store.record_menu_sync_failure(store_id, url, f"{type(exc).__name__}: {exc}")
        except Exception:
            log.exception("Unable to save menu sync failure status")
        return {"status": "failed", "store_id": store_id, "source_url": url,
                "error": f"{type(exc).__name__}: {exc}"}
