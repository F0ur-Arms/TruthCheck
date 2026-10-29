"""Small SQLite document store so review items and traces survive a restart."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DATA_DIR


class DocumentStore:
    def __init__(self, table: str, path: Optional[str] = None):
        path = path or os.environ.get("TRUTHCHECK_STORE_DB", str(Path(DATA_DIR) / "store.sqlite"))
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._table = table
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, doc TEXT NOT NULL)")
        self._conn.commit()

    def put(self, key: str, doc: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                f"INSERT INTO {self._table} (id, doc) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET doc = excluded.doc",
                (key, json.dumps(doc, default=str)),
            )
            self._conn.commit()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(f"SELECT doc FROM {self._table} WHERE id = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def all(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(f"SELECT doc FROM {self._table} ORDER BY rowid").fetchall()
        return [json.loads(r[0]) for r in rows]
