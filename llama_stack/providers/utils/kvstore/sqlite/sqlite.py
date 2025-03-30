# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from datetime import datetime
from typing import List, Optional

import aiosqlite

from ..api import KVStore
from ..config import SqliteKVStoreConfig


class SqliteKVStoreImpl(KVStore):
    def __init__(self, config: SqliteKVStoreConfig):
        self.db_path = config.db_path
        self.table_name = "kvstore"

    async def initialize(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expiration TIMESTAMP
                )
            """
            )
            await db.commit()

    async def set(self, key: str, value: str, expiration: Optional[datetime] = None) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"INSERT OR REPLACE INTO {self.table_name} (key, value, expiration) VALUES (?, ?, ?)",
                (key, value, expiration),
            )
            await db.commit()

    async def get(self, key: str) -> Optional[str]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(f"SELECT value, expiration FROM {self.table_name} WHERE key = ?", (key,)) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                value, expiration = row
                return value

    async def delete(self, key: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(f"DELETE FROM {self.table_name} WHERE key = ?", (key,))
            await db.commit()

    async def range(self, start_key: str, end_key: str) -> List[str]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"SELECT key, value, expiration FROM {self.table_name} WHERE key >= ? AND key <= ?",
                (start_key, end_key),
            ) as cursor:
                result = []
                async for row in cursor:
                    _, value, _ = row
                    result.append(value)
                return result
                
    async def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        async with aiosqlite.connect(self.db_path) as db:
            if prefix is None:
                async with db.execute(f"SELECT key FROM {self.table_name}") as cursor:
                    result = []
                    async for row in cursor:
                        result.append(row[0])
                    return result
            else:
                # SQLite doesn't have a native LIKE operator that works with parameters,
                # so we need to use the || operator to concatenate the prefix with '%'
                async with db.execute(
                    f"SELECT key FROM {self.table_name} WHERE key LIKE ? || '%'",
                    (prefix,),
                ) as cursor:
                    result = []
                    async for row in cursor:
                        result.append(row[0])
                    return result
