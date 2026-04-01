from wit_world.exports.cacheback import CacheTable as CacheTableBase
from lru_cache import TwoLevelLRUCache


class CacheTable(CacheTableBase):
    def __init__(self, leader_capacity, follower_capacity, leader_len, follower_len):
        self._cache = TwoLevelLRUCache(
            leader_capacity, follower_capacity, leader_len, follower_len
        )

    def update_cache(self, token_ids):
        total = self._cache._leader_len + self._cache._follower_len
        for i in range(len(token_ids) - total + 1):
            leader = tuple(token_ids[i : i + self._cache._leader_len])
            follower = tuple(
                token_ids[i + self._cache._leader_len : i + total]
            )
            self._cache.put(leader, follower)

    def get_draft_tokens(self, leader):
        key = tuple(leader[-self._cache._leader_len :])
        followers = self._cache.get(key)
        if followers is None:
            return []
        return [list(f) for f in followers]

    def clear(self):
        self._cache.clear()
