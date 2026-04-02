"""KV store interface implementation -- passthrough to host APIs."""

from typing import Optional

from wit_world.imports import kvs as _kvs


class Kvstore:
    def store_get(self, key: str) -> Optional[str]:
        return _kvs.store_get(key)

    def store_set(self, key: str, value: str) -> None:
        _kvs.store_set(key, value)

    def store_delete(self, key: str) -> None:
        _kvs.store_delete(key)

    def store_exists(self, key: str) -> bool:
        return _kvs.store_exists(key)

    def store_list_keys(self) -> list[str]:
        return list(_kvs.store_list_keys())
