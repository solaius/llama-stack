# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Optional

from .api import KVStore
from .config import KVStoreConfig, KVStoreType


def kvstore_dependencies():
    return ["aiosqlite", "psycopg2-binary", "redis", "pymongo"]


class InmemoryKVStoreImpl(KVStore):
    def __init__(self):
        self._store = {}

    async def initialize(self) -> None:
        pass

    async def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    async def set(self, key: str, value: str, expiration=None) -> None:
        self._store[key] = value

    async def delete(self, key: str) -> None:
        if key in self._store:
            del self._store[key]

    async def range(self, start_key: str, end_key: str) -> List[str]:
        return [self._store[key] for key in self._store.keys() if key >= start_key and key < end_key]
        
    async def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        if prefix is None:
            return list(self._store.keys())
        return [key for key in self._store.keys() if key.startswith(prefix)]


async def kvstore_impl(config: KVStoreConfig) -> KVStore:
    if config.type == KVStoreType.redis.value:
        from .redis import RedisKVStoreImpl

        impl = RedisKVStoreImpl(config)
    elif config.type == KVStoreType.sqlite.value:
        from .sqlite import SqliteKVStoreImpl

        impl = SqliteKVStoreImpl(config)
    elif config.type == KVStoreType.postgres.value:
        from .postgres import PostgresKVStoreImpl

        impl = PostgresKVStoreImpl(config)
    elif config.type == KVStoreType.mongodb.value:
        from .mongodb import MongoDBKVStoreImpl

        impl = MongoDBKVStoreImpl(config)
    else:
        raise ValueError(f"Unknown kvstore type {config.type}")

    await impl.initialize()
    return impl
